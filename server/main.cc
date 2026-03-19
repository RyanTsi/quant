#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

using json = nlohmann::json;

inline drogon::HttpResponsePtr newNlohmannJsonResponse(const json& j) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(j.dump());
    return resp;
}

struct DailyBar {
    std::string date;
    std::string symbol;
    double open, close, high, low, volume;
};

template<typename T>
struct DataBuffer {
public:
    // TODO: performance optimization
    void add(T&& item) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer.push_back(std::move(item));
    }
    
    std::vector<T> swap() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<T> temp;
        temp.swap(m_buffer);
        return temp;
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_buffer.empty();
    }
private:
    std::vector<T> m_buffer;
    mutable std::mutex m_mutex;
};

class MarketDataStorage {
public:
    MarketDataStorage() = default;

    void asyncSaveBatch(std::vector<DailyBar> batch) {
        if (batch.empty()) {
            return;
        }
        if (!drogon::app().isRunning()) {
            LOG_WARN << "Drogon app is not running yet, skipping this batch.";
            return;
        }
        if (!m_dbClient) {
            try {
                m_dbClient = drogon::app().getDbClient("default");
            } catch (const std::runtime_error &e) {
                LOG_ERROR << "Database client 'default' not found yet!";
                return; 
            }
        }
        std::ostringstream sql;
        sql << std::fixed << std::setprecision(6);
        sql << "INSERT INTO market_data_daily (date, symbol, open, high, low, close, volume) VALUES ";

        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& b = batch[i];
            sql << "('" << b.date << "', '" << b.symbol << "', " << b.open << ", " << b.high << ", " 
                << b.low << ", " << b.close << ", " << b.volume << ")";
            if (i != batch.size() - 1) {
                sql << ", ";
            }
        }
        sql << " ON CONFLICT (date, symbol) DO UPDATE SET "
            << "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
            << "close=EXCLUDED.close, volume=EXCLUDED.volume";
        
        m_dbClient->execSqlAsync(sql.str(),
            [count = batch.size()](const drogon::orm::Result& res){
                LOG_INFO << "Storage: Flushed " << count << " rows.";
            },
            [](const drogon::orm::DrogonDbException &e) mutable {
                LOG_ERROR << "Storage Error: " << e.base().what() << ". Re-queuing data...";
                // TODO: backup
            }
        );
    }
private:
    drogon::orm::DbClientPtr m_dbClient;
};


// TODO backup module
// class BackupManager {
// 
// }

class MarketDataManager {
public:
    void handleSingleIngest(const drogon::HttpRequestPtr& req, 
                      std::function<void (const drogon::HttpResponsePtr &)> &&callback) {
        try {
            auto j = json::parse(req->getBody());
            m_buffer.add(DailyBar{
                j.at("date"), j.at("symbol"), j.at("open"), j.at("close"), 
                j.at("high"), j.at("low"), j.at("volume")
            });
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setBody("{\"status\":\"success\"}\n");
            callback(resp);
        } catch (const std::exception& e) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setBody(std::string("{\"error\":\"") + e.what() + "\"}\n");
            callback(resp);
        }
    }

    void handleBatchIngest(const drogon::HttpRequestPtr& req, 
                      std::function<void (const drogon::HttpResponsePtr &)> &&callback) {
        try {
            auto j = json::parse(req->getBody());
            if (!j.is_array()) {
                throw std::runtime_error("Expected an array of daily bars");
            }
            for (const auto& item : j) {
                m_buffer.add(DailyBar{
                    item.at("date"), item.at("symbol"), item.at("open"), item.at("close"), 
                    item.at("high"), item.at("low"), item.at("volume")
                });
            }
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setBody("{\"status\":\"success\"}\n");
            callback(resp);
        } catch (const std::exception& e) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setBody(std::string("{\"error\":\"") + e.what() + "\"}\n");
            callback(resp);
        }
    }

    void flushToStorage(MarketDataStorage& storage) {
        if (m_buffer.empty()) {
            return;
        }
        auto full = m_buffer.swap();
        for(size_t i = 0; i < full.size(); i += max_batch_size) {
            size_t end_idx = std::min(i + max_batch_size, full.size());
            
            std::vector<DailyBar> batch(
                std::make_move_iterator(full.begin() + i),
                std::make_move_iterator(full.begin() + end_idx)
            );
            
            storage.asyncSaveBatch(std::move(batch));   
        }
    }

private:
    DataBuffer<DailyBar> m_buffer;
    size_t max_batch_size = 8192;
};

// 接口1：查询某天所有股票
void queryByDate(const drogon::HttpRequestPtr& req, 
                 std::function<void (const drogon::HttpResponsePtr &)> &&callback) {
    auto date = req->getParameter("date");
    if (date.empty()) {
        callback(newNlohmannJsonResponse(json{{"error", "missing date"}}));
        return;
    }

    auto client = drogon::app().getDbClient();
    client->execSqlAsync(
        "SELECT symbol, date, open, high, low, close, volume FROM market_data_daily WHERE date = $1",
        [callback](const drogon::orm::Result& res){
            json j = json::array();
            for (auto const& row : res) {
                j.push_back({
                    {"symbol", row["symbol"].as<std::string>()},
                    {"date", row["date"].as<std::string>()},
                    {"open", row["open"].as<double>()},
                    {"high", row["high"].as<double>()},
                    {"low", row["low"].as<double>()},
                    {"close", row["close"].as<double>()},
                    {"volume", row["volume"].as<double>()}
                });
            }
            callback(newNlohmannJsonResponse(j));
        },
        [callback](const drogon::orm::DrogonDbException &e){
            callback(newNlohmannJsonResponse(json{{"error", e.base().what()}}));
        },
        date
    );
}


// 接口2：查询某只股票在某段时间内的日线数据
void queryBySymbolAndDateRange(const drogon::HttpRequestPtr& req, 
                               std::function<void (const drogon::HttpResponsePtr &)> &&callback) {
    auto symbol = req->getParameter("symbol");
    auto start = req->getParameter("start_date");
    auto end = req->getParameter("end_date");

    if (symbol.empty() || start.empty() || end.empty()) {
        callback(newNlohmannJsonResponse(json{{"error", "missing parameters"}}));
        return;
    }

    auto client = drogon::app().getDbClient();
    client->execSqlAsync(
        "SELECT symbol, date, open, high, low, close, volume FROM market_data_daily "
        "WHERE symbol = $1 AND date >= $2 AND date <= $3 ORDER BY date ASC",
        [callback](const drogon::orm::Result& res){
            json j = json::array();
            for (auto const& row : res) {
                j.push_back({
                    {"symbol", row["symbol"].as<std::string>()},
                    {"date", row["date"].as<std::string>()},
                    {"open", row["open"].as<double>()},
                    {"close", row["close"].as<double>()},
                    {"high", row["high"].as<double>()},
                    {"low", row["low"].as<double>()},
                    {"volume", row["volume"].as<double>()}
                });
            }
            callback(newNlohmannJsonResponse(j));
        },
        [callback](const drogon::orm::DrogonDbException &e){
            callback(newNlohmannJsonResponse(json{{"error", e.base().what()}}));
        },
        symbol, start, end
    );
}

// TODO 接口3：查询某些股票在某段时间内的日线数据


int main() {
    drogon::app().loadConfigFile("config.json");
    
    static MarketDataManager manager;
    static MarketDataStorage storage;

    drogon::app().registerHandler("/api/v1/ingest/daily", [](const drogon::HttpRequestPtr& req, 
        std::function<void (const drogon::HttpResponsePtr &)> &&callback) {
        manager.handleBatchIngest(req, std::move(callback));
    }, {drogon::Post});

    // GET /api/v1/query/daily/all?date=2023-01-01
    drogon::app().registerHandler("/api/v1/query/daily/all", &queryByDate, {drogon::Get});

    // 接口2: GET /api/v1/query/daily/symbol?symbol=AAPL&start_date=2023-01-01&end_date=2023-12-31
    drogon::app().registerHandler("/api/v1/query/daily/symbol", &queryBySymbolAndDateRange, {drogon::Get});

    drogon::app().registerBeginningAdvice([](){
        LOG_INFO << "Drogon has started, initiating flush timer...";
        drogon::app().getLoop()->runEvery(5.0, [](){
            manager.flushToStorage(storage);
        });
    });

    LOG_INFO << "Quant Gateway running on http://0.0.0.0:8080";
    
    drogon::app().run();

    return 0;
}
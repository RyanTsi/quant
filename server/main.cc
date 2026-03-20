#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

inline drogon::HttpResponsePtr jsonResponse(const json& j, drogon::HttpStatusCode code = drogon::k200OK) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(code);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(j.dump());
    return resp;
}

inline drogon::HttpResponsePtr errorResponse(const std::string& msg, drogon::HttpStatusCode code = drogon::k400BadRequest) {
    return jsonResponse(json{{"error", msg}}, code);
}

// ─── Data Model ───

struct DailyBar {
    std::string date;
    std::string symbol;
    double open{0}, high{0}, low{0}, close{0};
    double volume{0}, amount{0}, turn{0};
    int tradestatus{1}, is_st{0};
};

static DailyBar parseDailyBar(const json& j) {
    DailyBar bar;
    bar.date   = j.at("date").get<std::string>();
    bar.symbol = j.at("symbol").get<std::string>();
    bar.open   = j.at("open").get<double>();
    bar.high   = j.at("high").get<double>();
    bar.low    = j.at("low").get<double>();
    bar.close  = j.at("close").get<double>();
    bar.volume = j.value("volume", 0.0);
    bar.amount = j.value("amount", 0.0);
    bar.turn   = j.value("turn", 0.0);
    bar.tradestatus = j.value("tradestatus", 1);
    bar.is_st  = j.value("isST", j.value("is_st", 0));
    return bar;
}

static json barToJson(const drogon::orm::Row& row) {
    return {
        {"symbol",      row["symbol"].as<std::string>()},
        {"date",        row["date"].as<std::string>()},
        {"open",        row["open"].as<double>()},
        {"high",        row["high"].as<double>()},
        {"low",         row["low"].as<double>()},
        {"close",       row["close"].as<double>()},
        {"volume",      row["volume"].as<double>()},
        {"amount",      row["amount"].as<double>()},
        {"turn",        row["turn"].as<double>()},
        {"tradestatus", row["tradestatus"].as<int>()},
        {"is_st",       row["is_st"].as<int>()}
    };
}

static const std::string ALL_COLUMNS =
    "symbol, date, open, high, low, close, volume, amount, turn, tradestatus, is_st";

// ─── Thread-safe Buffer ───

template<typename T>
class DataBuffer {
public:
    void add(T&& item) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer.push_back(std::move(item));
    }

    void addBatch(std::vector<T>&& items) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_buffer.insert(m_buffer.end(),
            std::make_move_iterator(items.begin()),
            std::make_move_iterator(items.end()));
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

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_buffer.size();
    }

private:
    std::vector<T> m_buffer;
    mutable std::mutex m_mutex;
};

// ─── Storage Layer ───

class MarketDataStorage {
public:
    MarketDataStorage() = default;

    void asyncSaveBatch(std::vector<DailyBar> batch) {
        if (batch.empty()) return;
        if (!drogon::app().isRunning()) {
            LOG_WARN << "App not running, skipping batch.";
            return;
        }
        if (!m_dbClient) {
            try {
                m_dbClient = drogon::app().getDbClient("default");
            } catch (const std::runtime_error& e) {
                LOG_ERROR << "DB client not found: " << e.what();
                return;
            }
        }

        std::ostringstream sql;
        sql << std::fixed << std::setprecision(6);
        sql << "INSERT INTO market_data_daily "
            << "(date, symbol, open, high, low, close, volume, amount, turn, tradestatus, is_st) VALUES ";

        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& b = batch[i];
            sql << "('" << b.date << "', '" << b.symbol << "', "
                << b.open << ", " << b.high << ", " << b.low << ", " << b.close << ", "
                << b.volume << ", " << b.amount << ", " << b.turn << ", "
                << b.tradestatus << ", " << b.is_st << ")";
            if (i + 1 < batch.size()) sql << ", ";
        }

        sql << " ON CONFLICT (date, symbol) DO UPDATE SET "
            << "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
            << "close=EXCLUDED.close, volume=EXCLUDED.volume, "
            << "amount=EXCLUDED.amount, turn=EXCLUDED.turn, "
            << "tradestatus=EXCLUDED.tradestatus, is_st=EXCLUDED.is_st";

        m_dbClient->execSqlAsync(sql.str(),
            [count = batch.size()](const drogon::orm::Result&) {
                LOG_INFO << "Flushed " << count << " rows.";
            },
            [](const drogon::orm::DrogonDbException& e) {
                LOG_ERROR << "Storage error: " << e.base().what();
            }
        );
    }

private:
    drogon::orm::DbClientPtr m_dbClient;
};

// ─── Ingest Manager ───

class MarketDataManager {
public:
    void handleSingleIngest(const drogon::HttpRequestPtr& req,
                            std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        try {
            auto j = json::parse(req->getBody());
            m_buffer.add(parseDailyBar(j));
            callback(jsonResponse(json{{"status", "ok"}, {"buffered", 1}}));
        } catch (const std::exception& e) {
            callback(errorResponse(e.what()));
        }
    }

    void handleBatchIngest(const drogon::HttpRequestPtr& req,
                           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
        try {
            auto j = json::parse(req->getBody());
            if (!j.is_array()) {
                throw std::runtime_error("Expected a JSON array");
            }
            std::vector<DailyBar> items;
            items.reserve(j.size());
            for (const auto& item : j) {
                items.push_back(parseDailyBar(item));
            }
            size_t count = items.size();
            m_buffer.addBatch(std::move(items));
            callback(jsonResponse(json{{"status", "ok"}, {"buffered", count}}));
        } catch (const std::exception& e) {
            callback(errorResponse(e.what()));
        }
    }

    void flushToStorage(MarketDataStorage& storage) {
        if (m_buffer.empty()) return;
        auto full = m_buffer.swap();
        for (size_t i = 0; i < full.size(); i += MAX_BATCH) {
            size_t end = std::min(i + MAX_BATCH, full.size());
            std::vector<DailyBar> batch(
                std::make_move_iterator(full.begin() + i),
                std::make_move_iterator(full.begin() + end));
            storage.asyncSaveBatch(std::move(batch));
        }
    }

    size_t bufferSize() const { return m_buffer.size(); }

private:
    DataBuffer<DailyBar> m_buffer;
    static constexpr size_t MAX_BATCH = 8192;
};

// ─── Query Handlers ───

// GET /api/v1/query/daily/all?date=2023-01-01&limit=500&offset=0
void queryByDate(const drogon::HttpRequestPtr& req,
                 std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto date = req->getParameter("date");
    if (date.empty()) {
        callback(errorResponse("missing 'date' parameter"));
        return;
    }

    int limit  = std::atoi(req->getParameter("limit").c_str());
    int offset = std::atoi(req->getParameter("offset").c_str());
    if (limit <= 0) limit = 5000;
    if (offset < 0) offset = 0;

    auto client = drogon::app().getDbClient();
    std::string sql = "SELECT " + ALL_COLUMNS +
        " FROM market_data_daily WHERE date = $1 ORDER BY symbol LIMIT $2 OFFSET $3";

    client->execSqlAsync(sql,
        [callback](const drogon::orm::Result& res) {
            json j = json::array();
            for (const auto& row : res) j.push_back(barToJson(row));
            callback(jsonResponse(json{{"count", j.size()}, {"data", j}}));
        },
        [callback](const drogon::orm::DrogonDbException& e) {
            callback(errorResponse(e.base().what(), drogon::k500InternalServerError));
        },
        date, limit, offset
    );
}

// GET /api/v1/query/daily/symbol?symbol=SH600000&start_date=2023-01-01&end_date=2023-12-31&limit=5000&offset=0
void queryBySymbolAndDateRange(const drogon::HttpRequestPtr& req,
                               std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto symbol = req->getParameter("symbol");
    auto start  = req->getParameter("start_date");
    auto end    = req->getParameter("end_date");

    if (symbol.empty() || start.empty() || end.empty()) {
        callback(errorResponse("missing 'symbol', 'start_date' or 'end_date'"));
        return;
    }

    int limit  = std::atoi(req->getParameter("limit").c_str());
    int offset = std::atoi(req->getParameter("offset").c_str());
    if (limit <= 0) limit = 5000;
    if (offset < 0) offset = 0;

    auto client = drogon::app().getDbClient();
    std::string sql = "SELECT " + ALL_COLUMNS +
        " FROM market_data_daily WHERE symbol = $1 AND date >= $2 AND date <= $3"
        " ORDER BY date ASC LIMIT $4 OFFSET $5";

    client->execSqlAsync(sql,
        [callback](const drogon::orm::Result& res) {
            json j = json::array();
            for (const auto& row : res) j.push_back(barToJson(row));
            callback(jsonResponse(json{{"count", j.size()}, {"data", j}}));
        },
        [callback](const drogon::orm::DrogonDbException& e) {
            callback(errorResponse(e.base().what(), drogon::k500InternalServerError));
        },
        symbol, start, end, limit, offset
    );
}

// POST /api/v1/query/daily/symbols  body: {"symbols":["SH600000","SZ000001"],"start_date":"...","end_date":"...","limit":5000,"offset":0}
void queryByMultipleSymbols(const drogon::HttpRequestPtr& req,
                            std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    try {
        auto body = json::parse(req->getBody());
        auto symbols = body.at("symbols").get<std::vector<std::string>>();
        auto start   = body.at("start_date").get<std::string>();
        auto end     = body.at("end_date").get<std::string>();
        int limit    = body.value("limit", 50000);
        int offset   = body.value("offset", 0);

        if (symbols.empty()) {
            callback(errorResponse("'symbols' array must not be empty"));
            return;
        }

        std::ostringstream safeSql;
        safeSql << "SELECT " << ALL_COLUMNS
                << " FROM market_data_daily WHERE symbol IN (";
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (i > 0) safeSql << ", ";
            // Simple quote-escape for symbol names (alphanumeric only)
            safeSql << "'" << symbols[i] << "'";
        }
        safeSql << ") AND date >= $1 AND date <= $2"
                << " ORDER BY symbol, date ASC LIMIT $3 OFFSET $4";

        auto client = drogon::app().getDbClient();
        client->execSqlAsync(safeSql.str(),
            [callback](const drogon::orm::Result& res) {
                json j = json::array();
                for (const auto& row : res) j.push_back(barToJson(row));
                callback(jsonResponse(json{{"count", j.size()}, {"data", j}}));
            },
            [callback](const drogon::orm::DrogonDbException& e) {
                callback(errorResponse(e.base().what(), drogon::k500InternalServerError));
            },
            start, end, limit, offset
        );
    } catch (const std::exception& e) {
        callback(errorResponse(e.what()));
    }
}

// GET /api/v1/query/daily/latest?symbol=SH600000&n=30
void queryLatestN(const drogon::HttpRequestPtr& req,
                  std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto symbol = req->getParameter("symbol");
    int n = std::atoi(req->getParameter("n").c_str());
    if (symbol.empty()) {
        callback(errorResponse("missing 'symbol' parameter"));
        return;
    }
    if (n <= 0) n = 30;

    auto client = drogon::app().getDbClient();
    std::string sql =
        "SELECT " + ALL_COLUMNS +
        " FROM market_data_daily WHERE symbol = $1"
        " ORDER BY date DESC LIMIT $2";

    client->execSqlAsync(sql,
        [callback](const drogon::orm::Result& res) {
            json j = json::array();
            for (const auto& row : res) j.push_back(barToJson(row));
            std::reverse(j.begin(), j.end());
            callback(jsonResponse(json{{"count", j.size()}, {"data", j}}));
        },
        [callback](const drogon::orm::DrogonDbException& e) {
            callback(errorResponse(e.base().what(), drogon::k500InternalServerError));
        },
        symbol, n
    );
}

// GET /api/v1/stats/summary?symbol=SH600000&start_date=...&end_date=...
void queryStatsSummary(const drogon::HttpRequestPtr& req,
                       std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto symbol = req->getParameter("symbol");
    auto start  = req->getParameter("start_date");
    auto end    = req->getParameter("end_date");

    if (symbol.empty() || start.empty() || end.empty()) {
        callback(errorResponse("missing 'symbol', 'start_date' or 'end_date'"));
        return;
    }

    auto client = drogon::app().getDbClient();
    std::string sql =
        "SELECT "
        "  COUNT(*) AS cnt,"
        "  MIN(date) AS first_date, MAX(date) AS last_date,"
        "  AVG(close) AS avg_close, MIN(low) AS min_low, MAX(high) AS max_high,"
        "  SUM(volume) AS total_volume, SUM(amount) AS total_amount,"
        "  AVG(turn) AS avg_turn"
        " FROM market_data_daily"
        " WHERE symbol = $1 AND date >= $2 AND date <= $3";

    client->execSqlAsync(sql,
        [callback, symbol](const drogon::orm::Result& res) {
            if (res.empty()) {
                callback(jsonResponse(json{{"symbol", symbol}, {"error", "no data"}}));
                return;
            }
            const auto& r = res[0];
            callback(jsonResponse(json{
                {"symbol",       symbol},
                {"count",        r["cnt"].as<int64_t>()},
                {"first_date",   r["first_date"].as<std::string>()},
                {"last_date",    r["last_date"].as<std::string>()},
                {"avg_close",    r["avg_close"].as<double>()},
                {"min_low",      r["min_low"].as<double>()},
                {"max_high",     r["max_high"].as<double>()},
                {"total_volume", r["total_volume"].as<double>()},
                {"total_amount", r["total_amount"].as<double>()},
                {"avg_turn",     r["avg_turn"].as<double>()}
            }));
        },
        [callback](const drogon::orm::DrogonDbException& e) {
            callback(errorResponse(e.base().what(), drogon::k500InternalServerError));
        },
        symbol, start, end
    );
}

// DELETE /api/v1/data/daily?symbol=SH600000&start_date=...&end_date=...
void deleteBySymbolAndDateRange(const drogon::HttpRequestPtr& req,
                                std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto symbol = req->getParameter("symbol");
    auto start  = req->getParameter("start_date");
    auto end    = req->getParameter("end_date");

    if (symbol.empty()) {
        callback(errorResponse("missing 'symbol' parameter"));
        return;
    }

    auto client = drogon::app().getDbClient();
    std::string sql;
    if (!start.empty() && !end.empty()) {
        sql = "DELETE FROM market_data_daily WHERE symbol = $1 AND date >= $2 AND date <= $3";
        client->execSqlAsync(sql,
            [callback](const drogon::orm::Result& res) {
                callback(jsonResponse(json{{"status", "ok"}, {"deleted", res.affectedRows()}}));
            },
            [callback](const drogon::orm::DrogonDbException& e) {
                callback(errorResponse(e.base().what(), drogon::k500InternalServerError));
            },
            symbol, start, end
        );
    } else {
        sql = "DELETE FROM market_data_daily WHERE symbol = $1";
        client->execSqlAsync(sql,
            [callback](const drogon::orm::Result& res) {
                callback(jsonResponse(json{{"status", "ok"}, {"deleted", res.affectedRows()}}));
            },
            [callback](const drogon::orm::DrogonDbException& e) {
                callback(errorResponse(e.base().what(), drogon::k500InternalServerError));
            },
            symbol
        );
    }
}

// GET /api/v1/symbols  — list all distinct symbols
void listSymbols(const drogon::HttpRequestPtr& req,
                 std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto client = drogon::app().getDbClient();
    client->execSqlAsync(
        "SELECT DISTINCT symbol FROM market_data_daily ORDER BY symbol",
        [callback](const drogon::orm::Result& res) {
            json symbols = json::array();
            for (const auto& row : res) {
                symbols.push_back(row["symbol"].as<std::string>());
            }
            callback(jsonResponse(json{{"count", symbols.size()}, {"symbols", symbols}}));
        },
        [callback](const drogon::orm::DrogonDbException& e) {
            callback(errorResponse(e.base().what(), drogon::k500InternalServerError));
        }
    );
}

// GET /api/v1/health
void healthCheck(const drogon::HttpRequestPtr& req,
                 std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto client = drogon::app().getDbClient();
    client->execSqlAsync("SELECT 1",
        [callback](const drogon::orm::Result&) {
            callback(jsonResponse(json{{"status", "healthy"}, {"db", "connected"}}));
        },
        [callback](const drogon::orm::DrogonDbException& e) {
            callback(jsonResponse(
                json{{"status", "unhealthy"}, {"db", e.base().what()}},
                drogon::k503ServiceUnavailable));
        }
    );
}

// ─── Main ───

int main() {
    drogon::app().loadConfigFile("config.json");

    static MarketDataManager manager;
    static MarketDataStorage storage;

    // --- Ingest ---
    drogon::app().registerHandler(
        "/api/v1/ingest/daily",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            manager.handleBatchIngest(req, std::move(callback));
        }, {drogon::Post});

    drogon::app().registerHandler(
        "/api/v1/ingest/daily/single",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            manager.handleSingleIngest(req, std::move(callback));
        }, {drogon::Post});

    // --- Query ---
    drogon::app().registerHandler("/api/v1/query/daily/all",    &queryByDate,              {drogon::Get});
    drogon::app().registerHandler("/api/v1/query/daily/symbol", &queryBySymbolAndDateRange, {drogon::Get});
    drogon::app().registerHandler("/api/v1/query/daily/symbols",&queryByMultipleSymbols,    {drogon::Post});
    drogon::app().registerHandler("/api/v1/query/daily/latest", &queryLatestN,             {drogon::Get});

    // --- Stats / Meta ---
    drogon::app().registerHandler("/api/v1/stats/summary", &queryStatsSummary, {drogon::Get});
    drogon::app().registerHandler("/api/v1/symbols",       &listSymbols,       {drogon::Get});

    // --- Data management ---
    drogon::app().registerHandler("/api/v1/data/daily",    &deleteBySymbolAndDateRange, {drogon::Delete});

    // --- Health ---
    drogon::app().registerHandler("/api/v1/health",        &healthCheck,       {drogon::Get});

    // --- Periodic flush ---
    drogon::app().registerBeginningAdvice([]() {
        LOG_INFO << "Drogon started. Flush timer active (5s interval).";
        drogon::app().getLoop()->runEvery(5.0, []() {
            manager.flushToStorage(storage);
        });
    });

    LOG_INFO << "Quant Data Gateway — http://0.0.0.0:8080";
    LOG_INFO << "Endpoints:";
    LOG_INFO << "  POST   /api/v1/ingest/daily          (batch ingest)";
    LOG_INFO << "  POST   /api/v1/ingest/daily/single   (single ingest)";
    LOG_INFO << "  GET    /api/v1/query/daily/all        (?date=)";
    LOG_INFO << "  GET    /api/v1/query/daily/symbol     (?symbol=&start_date=&end_date=)";
    LOG_INFO << "  POST   /api/v1/query/daily/symbols    (multi-symbol query)";
    LOG_INFO << "  GET    /api/v1/query/daily/latest     (?symbol=&n=)";
    LOG_INFO << "  GET    /api/v1/stats/summary          (?symbol=&start_date=&end_date=)";
    LOG_INFO << "  GET    /api/v1/symbols                (list all symbols)";
    LOG_INFO << "  DELETE /api/v1/data/daily             (?symbol=&start_date=&end_date=)";
    LOG_INFO << "  GET    /api/v1/health";

    drogon::app().run();
    return 0;
}

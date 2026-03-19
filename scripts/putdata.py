import requests
import os
from config.settings import settings
import utils.io

if __name__ == "__main__":
    url = "http://172.20.151.202:8080/"
    print("Starting data ingestion...")
    stock_data_dir = os.path.join(settings.data_path, "20100101-20260309")
    index_data_dir = os.path.join(settings.data_path, "cn_index")
    for filename in os.listdir(index_data_dir):
        df = utils.io.read_file_lines(os.path.join(index_data_dir, filename))
        map = df[0].split(",")
        map = {k: v for v, k in enumerate(map)}
        data_list = []
        for line in df[1:]:  # Skip header
            data = line.split(",")
            payload = {
                "date": data[map["date"]],
                "symbol": filename.split(".")[0],
                "open": float(data[map["open"]]),
                "high": float(data[map["high"]]),
                "low": float(data[map["low"]]),
                "close": float(data[map["close"]]),
                "volume": float(data[map["volume"]])
            }
            data_list.append(payload)
        for i in range(0, len(data_list), 4096):  # Batch size of 4096
            batch = data_list[i:i+4096]
            response = requests.post(url + "api/v1/ingest/daily", json=batch)
            if response.status_code == 200:
                print(f"Data for {filename.split('.')[0]} ingested successfully.")
            else:
                print(f"Failed to ingest data for {filename.split('.')[0]}. Status code: {response.status_code}, Response: {response.text}")    



import yaml
import json
from data_pipeline.loader import DataLoader

def main():
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    print(config)
    data_loader = DataLoader(config['data_path'])
    data_loader.fetch_current_stock_spot_df()

if __name__ == '__main__':
    main()

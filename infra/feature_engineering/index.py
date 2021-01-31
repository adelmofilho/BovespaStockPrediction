import json
import boto3 
import pandas as pd
from datetime import datetime 
from src.utils import upload_file, load_config
from src.feature_engineering import engineer_features


def lambda_handler(event, context):

    today = datetime.strftime(datetime.today(), "%Y%m%d")

    config = load_config() 
    window = config["feature_engineering"]["window"]
    scaler_config = config["feature_engineering"]["scaler"]

    s3 = boto3.client('s3')

    with open('/tmp/clean.csv', 'wb') as f:
        s3.download_fileobj('udacity-mle-capstone-977053370764-us-east-1', f"clean/{today}.csv", f)

    clean_data = pd.read_csv('/tmp/clean.csv')

    feature_table, scaler = engineer_features(clean_data, window, "predict", "IBOV", scaler_config=scaler_config)

    feature_table.to_csv("/tmp/feature.csv")   
    
    with open("/tmp/feature.csv", "rb") as f:
        s3.upload_fileobj(f, "udacity-mle-capstone-977053370764-us-east-1", f"processed/{today}.csv")
    
    return {
        'statusCode': 200,
        'body': today
    }

import json
import boto3 
import pandas as pd
from datetime import datetime 
from src.utils import upload_file, load_config
from src.data_preparation import prepare_data


def lambda_handler(event, context):

    today = datetime.strftime(datetime.today(), "%Y%m%d")

    config = load_config() 

    s3 = boto3.client('s3')

    with open('/tmp/raw.csv', 'wb') as f:
        s3.download_fileobj('udacity-mle-capstone-977053370764-us-east-1', f"raw/{today}.csv", f)

    raw_data = pd.read_csv('/tmp/raw.csv')

    clean_data = prepare_data(raw_data, split=None, split_valid=None, mode="predict")

    clean_data.to_csv("/tmp/clean.csv")   
    
    with open("/tmp/clean.csv", "rb") as f:
        s3.upload_fileobj(f, "udacity-mle-capstone-977053370764-us-east-1", f"clean_data/{today}.csv")
    
    return {
        'statusCode': 200,
        'body': today
    }

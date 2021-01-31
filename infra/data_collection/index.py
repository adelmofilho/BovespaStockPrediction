import json
import boto3 
from datetime import datetime 
from src.utils import upload_file, load_config
from src.data_collection import collect_data


def lambda_handler(event, context):

    today = datetime.strftime(datetime.today(), "%Y%m%d")

    config = load_config() 
    period = config["data_collection"]["period"]
    stocks = config["data_collection"]["stocks"]

    raw_data = collect_data(stocks=stocks, data_size=period)

    raw_data.to_csv("/tmp/ibovespa.csv")
   
    s3 = boto3.client('s3')
    with open("/tmp/ibovespa.csv", "rb") as f:
        s3.upload_fileobj(f, "udacity-mle-capstone-977053370764-us-east-1", f"raw/{today}.csv")
    
    return {
        'statusCode': 200,
        'body': today
    }

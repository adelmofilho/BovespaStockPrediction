import json
import boto3 
from yahooquery import Ticker
from datetime import datetime, timedelta
from src.utils import upload_file


def lambda_handler(event, context):

    today = datetime.strftime(datetime.today(), "%Y-%m-%d")
    tomorrow = datetime.strftime(datetime.today() + timedelta(days=1), "%Y-%m-%d")
    
    ibov = Ticker(symbols = "^BVSP")
    ibov_data = ibov.history(start=today, end=tomorrow).reset_index()
    ibov_data.to_csv("/tmp/ibovespa.csv")
   
    s3 = boto3.client('s3')
    with open("/tmp/ibovespa.csv", "rb") as f:
        s3.upload_fileobj(f, "udacity-capstone-977053370764", f"{today}.csv")
    
    return {
        'statusCode': 200,
        'body': today
    }

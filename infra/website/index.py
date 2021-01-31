import json
import boto3
from datetime import datetime, timedelta
import ast
import io
import pandas as pd
import plotly.express as px


def lambda_handler(event, context):
    
    s3 = boto3.resource('s3')
    obj = s3.Object("udacity-capstone-977053370764-us-east-1", "output/data.csv.out")
    body = obj.get()['Body'].read().decode("utf-8")
    prediction = ast.literal_eval(body)[0]
    
    
    obj = s3.Object("udacity-capstone-977053370764-us-east-1", "input/data.csv")
    body = obj.get()['Body'].read().decode("utf-8")
    
    dados = pd.read_csv(io.StringIO(body), sep=",")
    
    tomorrow = datetime.strftime(datetime.strptime(dados["date"][len(dados)-1], "%Y-%m-%d") + timedelta(days=1), "%Y-%m-%d")
    dados = dados[["date", "close"]]
    dados[["type"]] = "real"
    dados.loc[len(dados)] = [dados["date"][len(dados)-1], dados["close"][len(dados)-1], "forecast"]
    dados.loc[len(dados)] = [tomorrow, prediction, "forecast"]
        
    fig = px.scatter(dados, x='date', y="close", color="type", hover_data=['date', 'close'])
    fig.data[1].update(mode='markers+lines', hovertemplate="date=%{x}<br>close=%{y}<extra></extra>")
    fig.data[0].update(mode='markers+lines', hovertemplate="date=%{x}<br>close=%{y}<extra></extra>")
    fig.data = fig.data[::-1]
    fig.update_traces(marker=dict(size=12))
    fig.update_layout({'legend_title_text': ''})
    
    fig.write_html("/tmp/graph.html")
    
    s3_client = boto3.client('s3')
    response = s3_client.upload_file("/tmp/graph.html", "udacity-capstone-977053370764-us-east-1", "graph.html", ExtraArgs={'ContentType': 'text/html'})
    
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }

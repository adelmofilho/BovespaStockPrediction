# BovespaStockPrediction
## Machine Learning Engineer Nanodegree Capstone Project

### TL;DR

[![Foo](https://i.imgur.com/Moqw1IT.png)](http://udacity-capstone-977053370764-us-east-1.s3-website-us-east-1.amazonaws.com/)
<p align="center">Click on the image to go to the web application</p>

### Domain Background

Bovespa Index (Ibovespa) is one of the most important benchmark indices traded on the B3, stock exchange located in São Paulo, Brazil. Ibovespa takes into account around 80 stocks that comprehend brazilian companies from multiple sectors (financial, mining, oil & gas, electric utilities). These stocks are reviewed every four months, when their participation percentage can be modified (FINKLER, 2017).

Described as an indicator of the average performance of the most tradeable and representative assets of the Brazilian stock market (FARIA, 2012), Ibovespa fluctuations tend to represent important aspects of brazilian economy, such as foreign investments, monetary policy decisions and political issues.

### Problem Statement

For this project, a time series regression to predict the closing value for Bovespa index for the next trading day is proposed. 

In general terms, Bovespa index closing value can be defined as a function of its previous values (endogenous variables) and independent (exogenous) variables, for example, calendar variables (weekday, month), stock prices, dollar exchange.

Models of this nature provide useful support to decisions and allow simulations of different scenarios and the understanding of variables importance for the Bovespa index closing value.

## What makes this project special?

- A complete ETL pipeline was developed with AWS lambda functions and event bridge rules.
- AWS Sagemaker Batch Transformer to get inferences
    - Beyond the scope of this nanodegree, very common used on real life applications
    - Lower costs when compared with Sagemaker Endpoints.

### Solution Architecture

The final architecture differs positively from the one proposed on the capstone proposal:

- Serverless approach (less costs and maintenance)
- Event-driven architecture
- Easily update model

![](https://i.imgur.com/ApproIO.png)

Every day, around 9pm UTC-3, an event bridge rule triggers a lambda function that collects (raw) data from Yahoo finance API. Raw data is written on an AWS S3 bucket.

For each new raw data file, a S3 event notification triggers a lambda function to execute data preparation scripts. The same logic triggers a lambda function in order to create model features to get the model prediction.



### Model Architecture




### Capstone Proposal

Capstone proposal file is located at: [`docs/proposal.pdf`](https://github.com/adelmofilho/BovespaStockPrediction/blob/main/docs/proposal.pdf)
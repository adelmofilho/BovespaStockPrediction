# BovespaStockPrediction
### Machine Learning Engineer Nanodegree Capstone Project

## TL;DR

[![Foo](https://i.imgur.com/Moqw1IT.png)](http://udacity-capstone-977053370764-us-east-1.s3-website-us-east-1.amazonaws.com/)
<p align="center">Click on the image to go to the web application</p>

## Domain Background

Bovespa Index (Ibovespa) is one of the most important benchmark indices traded on the B3, stock exchange located in São Paulo, Brazil. Ibovespa takes into account around 80 stocks that comprehend brazilian companies from multiple sectors (financial, mining, oil & gas, electric utilities). These stocks are reviewed every four months, when their participation percentage can be modified (FINKLER, 2017).

Described as an indicator of the average performance of the most tradeable and representative assets of the Brazilian stock market (FARIA, 2012), Ibovespa fluctuations tend to represent important aspects of brazilian economy, such as foreign investments, monetary policy decisions and political issues.

## Problem Statement

For this project, a time series regression to predict the closing value for Bovespa index for the next trading day is proposed. 

In general terms, Bovespa index closing value can be defined as a function of its previous values (endogenous variables) and independent (exogenous) variables, for example, calendar variables (weekday, month), stock prices, dollar exchange.

Models of this nature provide useful support to decisions and allow simulations of different scenarios and the understanding of variables importance for the Bovespa index closing value.

## What makes this project special?

- A complete ETL pipeline was developed with AWS lambda functions and event bridge rules.
- AWS Sagemaker Batch Transformer to get inferences
    - Beyond the scope of this nanodegree, but very common used on real life applications
    - Lower costs when compared with Sagemaker Endpoints.

## Solution Architecture

*check out: [infra/](https://github.com/adelmofilho/BovespaStockPrediction/tree/main/infra)*

The final architecture differs positively from the one proposed on the capstone proposal:

- Serverless approach (lower costs and maintenance)
- Event-driven architecture
- Easily update model

![](https://i.imgur.com/ApproIO.png)

Every day, around 9pm UTC-3, an event bridge rule triggers a lambda function that collects (raw) data from Yahoo finance API. Raw data is written on an AWS S3 bucket.

For each new raw data file, a S3 event notification triggers a lambda function to execute data preparation scripts. The same logic triggers a lambda function in order to create model features to get the model prediction.

When a new feature table gets in on specified bucket, another lambda function starts a batch transformer job that gets a trained model and write the predictions on S3 bucket for a last lambda function to create the graph displayed on the website (public bucket associated with a route 53 DNS).

## Model Architecture

*check out: [ibovespa/model_training.py](https://github.com/adelmofilho/BovespaStockPrediction/blob/main/ibovespa/model_training.py)*

The purposed model architecture is composed of two full connected layer neural networks (fc) and a lstm neural network. Each element is related to one feature. Pytorch was the selected framework for this task.

![](https://i.imgur.com/6Wyf6Wt.png)

## Model evaluation

*check out: [LocalModelling.ipynb](https://github.com/adelmofilho/BovespaStockPrediction/blob/main/LocalModelling.ipynb)*

Model evaluation was performed on a test dataset (latest ~30 days) against a benchmark model (prediction equals to the last observed value) on the same dataset.

Mean absolute error (MAE) for both models were equivalent, but slightly lower on benchmark model. This performance is expected because ibovespa time series does not have expressive changes overnight.

However, low error is not the only propriety expected on this project. It is important that prediction have the correct sign of variation (if ibovespa will increase or decrease).

The benchmark model have F1-score of 0.5, completely random.

The model proposed for this project had a F1-Score ~0.6, i.e., not random.

The following plots represent (1st row) real (blue dots) and predict (red line) values of ibovespa; (2nd row) prediction versus observed values; and (3rd row) error histograms. All plotted using test dataset.

![](https://i.imgur.com/n5QqANA.png)

<p align="center"> Model proposed (left panels) and Benchmark model (right panels) evaluation plots</p>

</br>

## Capstone Proposal

Capstone proposal file is located at: [`docs/proposal.pdf`](https://github.com/adelmofilho/BovespaStockPrediction/blob/main/docs/proposal.pdf)
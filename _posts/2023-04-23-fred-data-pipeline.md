---
layout: post
title:  "Time Series Data Pipeline"
author: darsh
categories: [ ]
image: assets/images/fred-pipeline/fred-pipeline-architecture.png
---

##  Overview: Building a Scalable Time Series Data Pipeline

This project involved designing and implementing a robust data pipeline that collects monthly time series data from multiple sources, including FRED, US Census, alphavantage, and SEC Edgar, utilizing their respective APIs. __Airflow__, running within a __Docker container__, orchestrates the pipeline's logic as a single DAG. A Python operator within the DAG extracts data from each API and stores it in CSV format within an __S3 bucket__ acting as a data lake/landing zone. The pipeline guarantees the rate limit for each API is respected by limiting Airflow's maximum parallelism. After the data is stored within the S3 bucket, a __Lambda ETL task__ is initiated to read CSV files, apply necessary transformations, and upload the data to __AWS Timestream data warehouse__. Users can query time series data effortlessly utilizing SQL queries written on the AWS Query Editor. Moreover, we created a __Temporal Fusion Transformer (TFT) model__, which predicts inflation (CPI) in the US. This model is deployed on a local Jupyter notebook leveraging CUDA and can also be deployed on a SageMaker instance, enabling continuous data ingestion and prediction.


## Pipeline Orchestration: Streamlining Data Pipeline Management with Airflow on Docker

For orchestration, Airflow is a popular open-source platform for managing data pipelines. Its modular and scalable architecture allows users to easily manage and schedule complex workflows. Airflow provides a rich set of built-in operators and plugins for interfacing with a wide variety of data sources and destinations, making it a versatile tool for ETL and general data processing pipelines. Additionally, Airflow's web interface makes it easy to monitor and troubleshoot pipeline execution. However, Airflow can be complex to set up and configure, and scaling horizontally may require additional infrastructure resources. Additionally, users may need to develop custom operators or plugins to interface with certain data sources or destinations.

#### Setting up Airflow on Docker

Steps to setup airflow in docker:

1. Install Docker Community Edition (CE)
2. Install Docker Compose v1.29.1 or newer
3. run the following command to fetch the docker-compose.yaml file in terminal: 
  - `curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.6.0/docker-compose.yaml'`
4. Next, we must extend the official docker image in order to import our python dependencies. First, create a requirements.txt file and import the follwing packages:
  ```Text
  apache-airflow[amazon]
  apache-airflow-providers-http
  apache-airflow-providers-amazon
  matplotlib
  fredapi
  boto3
  setuptools
  awswrangler
  cryptography
  ```
5. Now, create a Dockerfile, and add the follwing code to the file:
```Dockerfile
FROM apache/airflow:2.5.3
COPY requirements.txt /requirements.txt
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt
  ```
6. Next run the follwing command in terminal:
   - `docker build . --tag extending_airflow:latest`
7. In the docker-compose.yml, under the webserver section, update image to:
```Dockerfile
webserver:
 image: extending_airflow:latest
```
8. Now you can simply run/shutdown the docker container using the follwoing commands
- `docker-compose up -d `
- `docker-compose down`
9. The default username/password is 'airflow', and the airflow ui can be accessed here: http://localhost:4000/fred-data-pipeline/

## Data Ingestion 

We implemented separate __Python operators__ to collect time series data from various APIs. These operators can be executed sequentially or in __parallel__ to efficiently fetch and process the data. Currently, we are running this pipeline __locally__ since it's a small task. However, to scale this pipeline, we can leverage the __modular architecture__ of Airflow to run these operators across multiple nodes in a distributed system. This can help improve __pipeline throughput__ and reduce the overall execution time.

#### Data Sources 

We are gathering monthly time series data that pertains to __demographic and economic__ factors. Our data sources include __FRED, US Census, alphavantage, and SEC Edgar__. The data we are collecting covers a range of economic indicators such as inflation rate, industrial production index, consumer price index, housing starts, crude oil prices, and employment data. In particular, we are focusing on selecting __leading indicators__ that have the potential to provide valuable information on future economic trends, specifically inflation.

#### Airflow DAG Setup

We have implemented separate python_callable files for each data source, which interact with their respective APIs, convert JSON responses to intermediate pandas dataframes, and write them to a CSV file in an S3 bucket. To avoid exceeding __API rate limits__ (set to 10), we utilize __Airflow pools__. These python_callable tasks may be executed in parallel, but for visual purposes, we run them sequentially as shown in the image below. Following every parallel execution of python operators, a task_#_check is called, which is a dummyoperator. This is necessary because Airflow does not permit [task_list] >> [task_list] notation. Instead, we must use [task_list] >> dummyOperator >> [task_list]. Finally, the lambda ETL job is triggered once all CSV files are written to the S3 bucket.

<img src="/assets/images/fred-pipeline/airflow_dag_graph.png" alt="description of image" width="900"/>

#### AWS S3: Landing zone for CSV files



## Lambda - ETL Job 

## AWS Timestream Setup 

---
layout: post
title:  "Data Pipeline for Time Series Data"
author: darsh
categories: [ ]
image: assets/images/fred-pipeline/fred-pipeline-architecture.png
---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview: Building a Scalable Time Series Data Pipeline](#overview-building-a-scalable-time-series-data-pipeline)
- [Pipeline Orchestration: Streamlining Data Pipeline Management with Airflow on Docker](#pipeline-orchestration-streamlining-data-pipeline-management-with-airflow-on-docker)
    - [Setting up Airflow on Docker](#setting-up-airflow-on-docker)
- [Data Ingestion](#data-ingestion)
    - [Data Sources](#data-sources)
    - [Airflow DAG Setup](#airflow-dag-setup)
    - [AWS S3: Landing zone for CSV files](#aws-s3-landing-zone-for-csv-files)
- [AWS Lambda: ETL Job](#aws-lambda-etl-job)
    - [AWS Lambda Setup](#aws-lambda-setup)
    - [ETL Overview: Batch Writing to Timestream](#etl-overview-batch-writing-to-timestream)
- [AWS Timestream: Time Series Data Warehouse](#aws-timestream-time-series-data-warehouse)
    - [AWS Timestream Setup](#aws-timestream-setup)
- [Temporal Fusion Transformer - Forecasting Inflation in US](#temporal-fusion-transformer---forecasting-inflation-in-us)


##  Overview: Building a Scalable Time Series Data Pipeline

This project involved designing and implementing a robust data pipeline that collects monthly time series data from multiple sources, including FRED, US Census, AlphaVantage, and SEC Edgar, utilizing their respective APIs. __Airflow__, running within a __Docker container__, orchestrates the pipeline's logic as a single DAG. A Python operator within the DAG extracts data from each API and stores it in CSV format within an __S3 bucket__ acting as a data lake/landing zone. The pipeline guarantees the rate limit for each API is respected by limiting Airflow's maximum parallelism. After the data is stored within the S3 bucket, a __Lambda ETL task__ is initiated to read CSV files, apply necessary transformations, and upload the data to __AWS Timestream data warehouse__. Users can query time series data effortlessly utilizing SQL queries written on the AWS Query Editor. Moreover, we created a __Temporal Fusion Transformer (TFT) model__, which predicts inflation (CPI) in the US. This model is deployed on a local Jupyter notebook leveraging CUDA and can also be deployed on a SageMaker instance, enabling continuous data ingestion and prediction.


## Pipeline Orchestration: Streamlining Data Pipeline Management with Airflow on Docker

For orchestration, __Airflow__ is a popular __open-source platform__ for managing data pipelines. Its __modular and scalable architecture__ allows users to easily manage and schedule complex workflows. Airflow provides a rich set of built-in operators and plugins for interfacing with a wide variety of data sources and destinations, making it a versatile tool for ETL and general data processing pipelines. Additionally, Airflow's __web interface__ makes it easy to monitor and troubleshoot pipeline execution. However, Airflow can be complex to set up and configure, and scaling horizontally may require additional infrastructure resources. Additionally, users may need to develop custom operators or plugins to interface with certain data sources or destinations.

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

The following code is for the python_callable for retrieving data from the third party API (AlphaVantage):

<html>
  <head>
    <style>
      pre {
        max-height: 300px;
        overflow-y: scroll;
        background-color: #f8f8f8;
        padding: 10px;
        border: 1px solid #ccc;
      }
    </style>
  </head>

  <pre class="code-block">
      <code class="language-python">
        import sys
        import pandas as pd
        from datetime import datetime
        sys.path.append('/opt/airflow/dags/common_package/')
        import config
        from fredapi import Fred
        import awswrangler as wr
        import boto3
        from cryptography.fernet import Fernet
        from airflow.models import Variable
        import requests


        # Define the AWS credentials and S3 bucket name
        #aws_access_key_id = config.get_aws_key_id()
        #aws_secret_access_key = config.get_secret_access_key()
        s3_bucket_name = config.get_s3_bucket()
        fernet_key = config.get_fernet_key()
        fred_api_key = config.get_fred_key()
        vantage_api_key = config.get_alpha_vantage_key()

        # Retrieve the encrypted credentials from Airflow Variables
        encrypted_access_key_id = Variable.get('aws_access_key_id', deserialize_json=False)
        encrypted_secret_access_key = Variable.get('aws_secret_access_key', deserialize_json=False)

        # Decrypt the credentials using the encryption key
        key = fernet_key  # Replace with the encryption key generated in step 1
        fernet = Fernet(key)
        aws_access_key_id = fernet.decrypt(encrypted_access_key_id.encode()).decode()
        aws_secret_access_key = fernet.decrypt(encrypted_secret_access_key.encode()).decode()


        # Define a function to fetch data for a given FRED series and store it in a CSV file in S3
        def fetch_vantage_series_to_s3(vantage_series_id, **context):
            # Make API call and convert fron pd series to dataframe
            vantage_series = get_vantage_series(vantage_series_id, "TIME_SERIES_MONTHLY", vantage_api_key)

            for i in range(2):
                df = vantage_series[i]
                df = df.to_frame().reset_index()
                df = df.rename(columns={0:'value'})
                
                s3_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)
                
                if i == 0: # for close price
                    # Generate a unique file name for the CSV file
                    file_name = f'{vantage_series_id.replace("/", "-")}-{datetime.now().strftime("%Y%m%d")}.csv'
                    # Write the DataFrame to a CSV file and upload it to S3
                    wr.s3.to_csv(df=df, path=f"s3://{s3_bucket_name}/csv-series-vantage/{file_name}", index=False, boto3_session=s3_session)
                else: # for volumne
                    # Generate a unique file name for the CSV file
                    name = f'{vantage_series_id}VOL'
                    file_name = f'{name.replace("/", "-")}-{datetime.now().strftime("%Y%m%d")}.csv'
                    # Write the DataFrame to a CSV file and upload it to S3
                    wr.s3.to_csv(df=df, path=f"s3://{s3_bucket_name}/csv-series-vantage/{file_name}", index=False, boto3_session=s3_session)


        def get_vantage_series(series, function, alpha_vantage_key):
            if function == "TIME_SERIES_MONTHLY":
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={series}&apikey={alpha_vantage_key}"
                r = requests.get(url)
                json_data = r.json()
                monthly_data = json_data["Monthly Time Series"]
                close_series = pd.Series(
                    {datetime.strptime(date, '%Y-%m-%d').replace(day=1): float(data["4. close"]) for date, data in monthly_data.items()},
                    name="Close")
                # Convert the index to a DatetimeIndex
                #print(type(close_series.index))
                #atetime_object = datetime.strptime(str(close_series.index), '%y-%m-%d').replace(day=1)
                #print(datetime_object)
                close_series.index = pd.to_datetime(close_series.index)
                
                vol_series = pd.Series(
                    {datetime.strptime(date, '%Y-%m-%d').replace(day=1): float(data["5. volume"]) for date, data in monthly_data.items()},
                    name="Volume")
                vol_series.index = pd.to_datetime(vol_series.index)
                
                
                close_series.name = series
                vol_series.name = series + "-VOL"
                
                #close_series.name = close_series.index.replace(day="01")
                #vol_series.name = vol_series.index.replace(day="01")
                
                data = [close_series, vol_series]
            elif function == "DIGITAL_CURRENCY_MONTHLY":
                url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_MONTHLY&symbol={series}&market=USD&apikey={alpha_vantage_key}'
                r = requests.get(url)
                json_data = r.json()
                #data = json_data
                monthly_data = json_data["Time Series (Digital Currency Monthly)"]
                close_series = pd.Series(
                    {datetime.strptime(date, '%Y-%m-%d').replace(day=1): float(data["4a. close (USD)"]) for date, data in monthly_data.items()},
                    name="Close")
                # Convert the index to a DatetimeIndex
                print(type(close_series.index))
                close_series.index = pd.to_datetime(close_series.index)
                
                vol_series = pd.Series(
                    {datetime.strptime(date, '%Y-%m-%d').replace(day=1): float(data['5. volume']) for date, data in monthly_data.items()},
                    name="Volume")
                vol_series.index = pd.to_datetime(vol_series.index)
                data = [close_series, vol_series]
            return data
      </code>
  </pre>
</html>

The code for our dag is available below. 

<html>
  <head>
    <style>
      pre {
        max-height: 300px;
        overflow-y: scroll;
        background-color: #f8f8f8;
        padding: 10px;
        border: 1px solid #ccc;
      }
    </style>
  </head>

  <pre class="code-block">
      <code class="language-python">
          from datetime import datetime
          import requests
          import datetime
          from airflow import DAG
          from airflow.operators.python_operator import PythonOperator
          from airflow.operators.dummy_operator import DummyOperator
          from airflow.hooks.S3_hook import S3Hook


          import sys
          sys.path.append('/opt/airflow/dags/common_package/')
          import config
          import fetch_fred_series_s3
          import states_met_to_s3
          import lambda_invoke
          import fetch_vantage_series_s3
          import fetch_census_series_s3

          def test_task():
            print("First dag task test: successful")

          # Define the DAG
          # to limit the number of proccess calling fred api (60/min) ~ we limit the execution parallelism to 10 tasks using airflow pools
          with DAG(
              'fred_to_s3_batch_dag',
              start_date=datetime.datetime(2023, 4, 10),
              schedule_interval='@once', #Batch upload
              catchup=False,
              default_args={
                  'owner': 'airflow',
                  'depends_on_past': False,
                  'email_on_failure': False,
                  'email_on_retry': False,
                  'retries': 2,
                  'retry_delay': datetime.timedelta(minutes=5),
                  'pool':'fred_batch_pool'
              }
          ) as dag:
            
            test_task1 = PythonOperator(
              task_id='test_to_s3_task1',
              python_callable=test_task,
            )


            states_met_to_s3_task_final = PythonOperator(
              task_id='states_met_to_s3_task_final',
              python_callable=states_met_to_s3.states_met_to_s3,
              op_kwargs={
                  'test': 'Last time this works',
              },
              provide_context=True,
            )

            lambda_invoke = PythonOperator(
              task_id='lambda_invoke',
              python_callable=lambda_invoke.lambda_invoke_test
            )

            task_1_check = DummyOperator(task_id='task_1_check')

            task_2_check = DummyOperator(task_id='task_2_check')

            task_3_check = DummyOperator(task_id='task_3_check')


          # Define a list of FRED series IDs to fetch
          fred_series_ids = ["CPIAUCNS", # Target
                        
                        "M2SL", "INDPRO", "PPIACO", "CPITRNSL", "POPTHM", "CES4300000001", "USEPUINDXM", "DSPIC96", # CONVERT TO GROWTH RATE

                        "HOUST", "MCOILWTICO", "FEDFUNDS", "UNRATE" # Keep as is
                      ]
          # Define a PythonOperator for each FRED series to fetch
          fred_series_tasks = []
          for fred_series_id in fred_series_ids:
              task_id = f'fetch_{fred_series_id}_to_s3'
              op = PythonOperator(
                  task_id=task_id,
                  python_callable=fetch_fred_series_s3.fetch_fred_series_to_s3,
                  op_kwargs={
                      'fred_series_id': fred_series_id,
                  },
                  provide_context=True
                  
              )
              fred_series_tasks.append(op)


          vantage_series_ids = ["VOO", "XLP", "XLE", "XLB", "XAG"] # voo = market trackings, xlp=consumer staples, xle = energy, xlb = basic materials, xag = gold
          vantage_series_tasks = []
          for vantage_series_id in vantage_series_ids:
              task_id_vantage = f'fetch_{vantage_series_id}_to_s3'
              op = PythonOperator(
                  task_id=task_id_vantage,
                  python_callable=fetch_vantage_series_s3.fetch_vantage_series_to_s3
          ,
                  op_kwargs={
                      'vantage_series_id': vantage_series_id,
                  },
                  provide_context=True
                  
              )
              vantage_series_tasks.append(op)

          vantage_series_ids = ["VOO"]


          census_series_ids = ["firms", "emp", "fsize1", "fsize2", 'taxes', "payroll", "sup_val"]
          census_series_tasks = []
          for census_series_id in census_series_ids:
              task_id_census = f'fetch_{census_series_id}_to_s3'
              op = PythonOperator(
                  task_id=task_id_census,
                  python_callable=fetch_census_series_s3.fetch_census_series_to_s3
          ,
                  op_kwargs={
                      'series_id': census_series_id,
                  },
                  provide_context=True
                  
              )
              census_series_tasks.append(op)


          # Set the task dependencies so that all FRED series are fetched in parallel
          test_task1  >> fred_series_tasks >> task_1_check >> vantage_series_tasks >> task_2_check >> census_series_tasks >> task_3_check >> states_met_to_s3_task_final >>  lambda_invoke
      </code>
  </pre>
</html>



#### AWS S3: Landing zone for CSV files

AWS S3 serves as our __landing zone__ for all the CSV files generated by the data ingestion pipeline. We created an S3 bucket in the __same region__ as our Airflow lambda instance for low latency access. This region choice is important as it can affect network latency and data transfer rates. In addition, we enabled versioning on the S3 bucket to ensure we can access previous versions of the CSV files if necessary.

Having a landing zone for our CSV files is an important component of a data pipeline as it allows us to store, access, and process large amounts of data efficiently. This is often referred to as a __"data lake" approach__, where data is stored in its raw form and processed as needed. An alternative approach is the __"ELT"__ (extract, load, transform) method, where data is extracted from its source, loaded into a centralized database, and then transformed into a usable format. While both approaches have their pros and cons, the data lake approach allows for greater flexibility and scalability as it enables processing of data in its raw form without the need for complex ETL processes.

Note that the S3 bucket name must be __unique globally__. Our bucket is called dk-airflow-test-bucket. Below we can see that a separate folder is created for each of the data sources.

<img src="/assets/images/fred-pipeline/s3_bucket_folders.png" alt="Image 1" style="width:500px;">

Next, we can see all the files that have been added to the folder for FRED, after a successful DAG run.

<img src="/assets/images/fred-pipeline/csv_series_fred_s3.png" alt="Image 2" style="width:500px;">

## AWS Lambda: ETL Job 

This __ETL (Extract, Transform, Load)__ job comprises a Python script that functions as an AWS Lambda triggered by Airflow following successful ingestion of data into S3. The job reads and transforms multiple CSV files from an S3 bucket, and then writes the data to Amazon Timestream, a fully managed time series database. Notably, if a Timestream Table already contains data, only missing or future values are inserted into the database. The following are the primary steps of this job:


#### AWS Lambda Setup

Lambda Setup:
- Memory: 128 MB, Ephemeral Storage: 512 MB, Timeout: 2 min (price per 1ms: $0.0000000083, __Free Tier Eligible__)
- We add a __layer__ to our lambda function to allow us to work with pandas (arn:aws:lambda:us-east-2:336392948345:layer:AWSSDKPandas-Python39:6).
- We assign an create an __IAM Role__ with full read/write access to Timestream and S3, and assign it to the lambda function

#### ETL Overview: Batch Writing to Timestream

ETL Steps:
- Define the S3 bucket, Timestream client, and the database and table names, utilizing the __boto3__ package for python to communicate with AWS services.
- Create a Timestream table with the create_timestream_table function, as detailed in the (#AWS Timestream Setup) section.
- __Extract:__ Read CSV files from each folder in the S3 bucket and merge them into a single Pandas dataframe through __joining on date_index.__
- __Transform:__ Convert pandas datetime to Unix timestamp and create a Timestream record if year >= 1970 (or has a __non-negative Unix__ value).
- __Load:__ Batch write to Timestream with a __batch size of 100 Timestream records__ to maximize throughput.

## AWS Timestream: Time Series Data Warehouse

AWS Timestream is a __fully managed time-series database service__ offered by Amazon Web Services (AWS) that is designed to handle large-scale time series data with high durability and availability. __Unlike traditional SQL databases, Timestream is optimized for handling time series data__ and provides features like automatic scaling, data retention management, and the ability to query large datasets quickly. One of the main advantages of Timestream over traditional databases is that it allows for efficient and easy storage and retrieval of time series data at scale.

Other populer alternatives to Timstream include __InfluxDB, OpenTSDB, and TimescaleDB__. These databases provide similar features to Timestream, but they may differ in terms of performance, scalability, and ease of use. InfluxDB, for instance, is a popular open-source time-series database that offers high write and query performance and supports SQL-like query languages. OpenTSDB is another popular open-source database that provides horizontal scaling and advanced features like histograms and percentile aggregations. TimescaleDB is an open-source database that extends PostgreSQL to handle time-series data and provides features like automatic data partitioning and multi-node clustering. However, Timestream is specifically designed and __optimized for the AWS ecosystem__, which makes it a better choice if you are already using AWS services and need a fully managed time-series database that can easily integrate with other AWS services.

#### AWS Timestream Setup

Our created Timestream table has the following properties:

- Database retention period: 73000 days for magnetic store (Max amount for historical data) and 24 hours for memory store
- Memory store and magnetic store writes are enabled
- Magnetic store rejected data is stored in an S3 bucket with Server-Side Encryption (SSE-S3)

## Temporal Fusion Transformer - Forecasting Inflation in US

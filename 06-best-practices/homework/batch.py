#!/usr/bin/env python
# coding: utf-8

from calendar import c
import sys
import pickle
import pandas as pd
import os

S3_ENDPOINT_URL = "http://localhost:4566/"

def get_input_path(year, month):
    default_input_pattern = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def main(year:str, month:str):

    input_file = get_input_path(year, month)
    print(input_file)
    output_file = get_output_path(year, month)
    print(output_file)
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    categorical = ['PUlocationID', 'DOlocationID']

    options = {'client_kwargs': {'endpoint_url': S3_ENDPOINT_URL},'key': 'abc', 'secret': 'xyz'}

    df = pd.read_parquet(input_file, storage_options=options)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file,
                        engine='pyarrow',
                        compression=None,
                        index=False,
                        storage_options=options)


def read_data(filename:str):
    
    df = pd.read_parquet(filename)
    return df

def prepare_data(df:pd.DataFrame, categorical:list):
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df






def run():

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)

if __name__ == '__main__':
    run()


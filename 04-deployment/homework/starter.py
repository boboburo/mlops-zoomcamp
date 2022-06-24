#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    return dicts

def load_model(filename):
    with open('model.bin', 'rb') as f_in:
            dv, lr = pickle.load(f_in)

    return dv, lr

def save_results(df, y_pred, model_name, output_file, year, month):
    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False)

#@task
def apply_model(input_file, model_file, output_file, year, month):
    #logger = get_run_logger()

    print(f'reading the data from {input_file}...')
    df = read_data(input_file)
    dicts = prepare_dictionaries(df)

    print(f'loading the model with RUN_ID={model_file}...')
    dv, lr = load_model(model_file)

    print(f'applying the model...')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f'mean prediction is {y_pred.mean()}')

    print(f'saving the result to {output_file}...')
    save_results(df, y_pred, model_file, output_file, year, month)

    return

def get_paths(year, month):
    input_file = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"homework4_df_results_{year:04d}-{month:02d}.parquet"

    return input_file, output_file

def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 2
    model_file = sys.argv[3] # 'model.bin'

    input_file, output_file = get_paths(year, month)
    
    apply_model(input_file, model_file, output_file, year, month)

if __name__ == '__main__':
    run()



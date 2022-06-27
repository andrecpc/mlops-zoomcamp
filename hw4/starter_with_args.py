import pickle
import sys

import pandas as pd


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr


categorical = ['PUlocationID', 'DOlocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def predict(year, month):
    df = read_data(
        f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    dv, lr = load_model()
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    return y_pred, df


def save_output(y_pred, df, year, month, taxi_type='fhv'):
    output_file = f'predictions_{year:04d}-{month:02d}.parquet'
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['pred'] = y_pred
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run(year: int, month: int):
    y_pred, df = predict(year, month)
    print(f'mean predicted duration: {y_pred.mean()}')
    save_output(y_pred, df, year, month)


if __name__ == '__main__':

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    run(year, month)

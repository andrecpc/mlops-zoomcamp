from batch import prepare_data
import pandas as pd
from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PUlocationID",
        "DOlocationID",
        "pickup_datetime",
        "dropOff_datetime",
        "duration",
    ]
    df = pd.DataFrame(data, columns=columns[:-1])

    output_data = [
        ["-1", "-1", dt(1, 2), dt(1, 10), 8.000000000000002],
        ["1", "1", dt(1, 2), dt(1, 10), 8.000000000000002],
    ]
    output_df = pd.DataFrame(output_data, columns=columns)

    categorical = ["PUlocationID", "DOlocationID"]

    assert prepare_data(df, categorical).equals(output_df)

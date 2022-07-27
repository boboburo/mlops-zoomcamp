import batch
import pandas as pd
from deepdiff import DeepDiff
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


test_data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
test_df = pd.DataFrame(test_data, columns=columns)   

expected_data = [
    (-1, -1, dt(1, 2), dt(1, 10),8),
    (1, 1, dt(1, 2), dt(1, 10),8)
]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime','duration']
expected_df = pd.DataFrame(expected_data, columns=columns)
expected_df[columns[0:2]] = expected_df[columns[0:2]].astype('str')
expected_df[columns[-1]] = expected_df[columns[-1]].astype('float')


def test_prepare_data():
    actual_df = batch.prepare_data(test_df, categorical= columns[0:2])
    
    diff = DeepDiff(actual_df.to_dict(), expected_df.to_dict(), significant_digits=0)
    print(f'diff={diff}')

    assert 'type_changes' not in diff
    assert 'values_changed' not in diff
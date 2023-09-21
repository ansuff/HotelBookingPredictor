import os
import sys

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import DataLoader, TrainTestSplitter


@pytest.fixture
def data():
    data = pd.read_csv(os.path.join("data", "hotel_bookings.csv"))
    data['arrival_date'] = pd.to_datetime(data['arrival_date'])
    return data


def test_data_loader():
    data_loader = DataLoader()
    df = data_loader.load_data(os.path.join("data", "hotel_bookings.csv"))
    assert isinstance(df, pd.DataFrame)


def test_train_test_splitter(data):
    train_test_splitter = TrainTestSplitter()
    X_train, X_test, y_train, y_test = train_test_splitter.split(data,encoded=False)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
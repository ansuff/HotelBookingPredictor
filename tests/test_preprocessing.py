import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import DataPreprocessor


@pytest.fixture
def data():
    data = pd.read_csv(os.path.join("data", "hotel_bookings.csv"))
    data['arrival_date'] = pd.to_datetime(data['arrival_date'])
    return data


def test_convert_date_columns_to_datetime(data):
    preprocessor = DataPreprocessor()
    data = preprocessor.convert_date_columns_to_datetime(data)
    assert isinstance(data['arrival_date'][0], pd.Timestamp)
    assert isinstance(data['departure_date'][0], pd.Timestamp)
    assert isinstance(data['booking_date'][0], pd.Timestamp)


def test_drop_na_rows(data):
    preprocessor = DataPreprocessor()
    data = preprocessor.drop_na_rows(data)
    assert data.isnull().sum().sum() == 0


def test_drop_duplicates(data):
    preprocessor = DataPreprocessor()
    data = preprocessor.drop_duplicates(data)
    assert data.duplicated().sum() == 0


def test_drop_negative_adr(data):
    preprocessor = DataPreprocessor()
    data = preprocessor.drop_negative_adr(data)
    assert (data['adr'] >= 0).all()


def test_fit_transform(data):
    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform(data)
    assert isinstance(data['arrival_date'][0], pd.Timestamp)
    assert isinstance(data['departure_date'][0], pd.Timestamp)
    assert isinstance(data['booking_date'][0], pd.Timestamp)
    assert data.isnull().sum().sum() == 0
    assert data.duplicated().sum() == 0
    assert (data['adr'] >= 0).all()
    assert data.select_dtypes(include='object').empty


def test_fit_transform_without_encoding_variables(data):
    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform_without_encoding_variables(data)
    assert isinstance(data['arrival_date'][0], pd.Timestamp)
    assert isinstance(data['departure_date'][0], pd.Timestamp)
    assert isinstance(data['booking_date'][0], pd.Timestamp)
    assert data.isnull().sum().sum() == 0
    assert data.duplicated().sum() == 0
    assert (data['adr'] >= 0).all()
    assert not data.select_dtypes(include='object').empty
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from feature_engineering import HotelBookingFeatures


@pytest.fixture
def data():
    data = pd.read_csv(os.path.join("data", "hotel_bookings.csv"))
    data['arrival_date'] = pd.to_datetime(data['arrival_date'])
    return data


def test_is_weekend_stay(data):
    feature_engineer = HotelBookingFeatures()
    data = feature_engineer.is_weekend_stay(data)
    assert data['is_weekend_stay'].isin([0, 1]).all()


def test_num_days_stayed(data):
    feature_engineer = HotelBookingFeatures()
    data = feature_engineer.num_days_stayed(data)
    assert (data['num_days_stayed'] >= 0).all()


def test_booking_lead_time(data):
    feature_engineer = HotelBookingFeatures()
    data = feature_engineer.booking_lead_time(data)
    assert (data['booking_lead_time'] >= 0).all()


def test_arrival_date_features(data):
    feature_engineer = HotelBookingFeatures()
    data = feature_engineer.arrival_date_features(data)
    assert data['arrival_dayofweek'].isin(range(7)).all()
    assert data['arrival_month'].isin(range(1, 13)).all()
    assert data['arrival_year'].isin([2015, 2016, 2017,2018]).all()
    assert data['arrival_weekofyear'].isin(range(0, 53)).all()
    assert data['arrival_date_month'].isin(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']).all()


def test_fit_transform(data):
    feature_engineer = HotelBookingFeatures()
    data['arrival_date'] = pd.to_datetime(data['arrival_date'])
    data = feature_engineer.fit_transform(data)
    assert data['is_weekend_stay'].isin([0, 1]).all()
    assert (data['num_days_stayed'] >= 0).all()
    assert (data['booking_lead_time'] >= 0).all()
    assert data['arrival_dayofweek'].isin(range(7)).all()
    assert data['arrival_month'].isin(range(1, 13)).all()
    assert data['arrival_year'].isin([2015, 2016, 2017,2018]).all()
    assert data['arrival_weekofyear'].isin(range(0, 53)).all()
    assert data['arrival_date_month'].isin(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']).all()
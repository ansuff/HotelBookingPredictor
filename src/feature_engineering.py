import pandas as pd

class HotelBookingFeatures:
    def __init__(self, data):
        self.data = data
    
    def is_weekend_stay(self):
        arrival_dayofweek = pd.DatetimeIndex(self.data['arrival_date']).dayofweek
        departure_dayofweek = pd.DatetimeIndex(self.data['departure_date']).dayofweek
        self.data['is_weekend_stay'] = ((arrival_dayofweek >= 5) | (departure_dayofweek >= 5)).astype(int)
            
    def num_days_stayed(self):
        self.data['num_days_stayed'] = (pd.DatetimeIndex(self.data['departure_date']) - pd.DatetimeIndex(self.data['arrival_date'])).days
    
    def booking_lead_time(self):
        self.data['booking_lead_time'] = (pd.DatetimeIndex(self.data['arrival_date']) - pd.DatetimeIndex(self.data['booking_date'])).days
import pandas as pd

class HotelBookingFeatures:
    '''
    This class contains the methods for creating new features from the data.
    
    The methods are:
    - is_weekend_stay
    - num_days_stayed
    - num_weekend_nights
    - num_week_nights
    - booking_lead_time
    - arrival_date_features
    
    The fit_transform method applies all the feature engineering steps to the data.
    You can eaiter use the fit_transform method or apply the methods one by one.
    '''
    def __init__(self):
        pass
    
    def is_weekend_stay(self, data):
        '''
        This method creates a new variable 'is_weekend_stay' from the arrival and departure dates.
        
        The value of 'is_weekend_stay' is 1 if the stay includes a weekend night.
        '''
        arrival_dayofweek = pd.DatetimeIndex(data['arrival_date']).dayofweek
        departure_dayofweek = pd.DatetimeIndex(data['departure_date']).dayofweek
        data['is_weekend_stay'] = ((arrival_dayofweek >= 5) | (departure_dayofweek >= 5)).astype(int)
        return data
            
    def num_days_stayed(self, data):
        '''
        This method creates a new variable 'num_days_stayed' from the arrival and departure dates.
        '''
        data['num_days_stayed'] = (pd.DatetimeIndex(data['departure_date']) - pd.DatetimeIndex(data['arrival_date'])).days
        return data
        '''
    def num_weekend_and_week_nights(self, data):
        '''
        #This method creates new variables 'num_weekend_nights' and 'num_week_nights' from the arrival and departure dates.
        '''
        # calculate the duration of the stay in days
        data['stay_duration'] = (data['departure_date'] - data['arrival_date']).dt.days

        # calculate the number of weekday and weekend days
        weekday_mask = (data['arrival_date'].dt.weekday < 5) & (data['departure_date'].dt.weekday < 5)
        weekend_mask = (data['arrival_date'].dt.weekday >= 5) | (data['departure_date'].dt.weekday >= 5)
        data['num_weekday_days'] = weekday_mask * data['stay_duration']
        data['num_weekend_days'] = weekend_mask * data['stay_duration']

        # drop the stay duration column
        data = data.drop('stay_duration', axis=1)

        return data
        '''          
    def booking_lead_time(self, data):
        '''
        This method creates a new variable 'booking_lead_time' from the arrival and booking dates.
        '''
        data['booking_lead_time'] = (pd.DatetimeIndex(data['arrival_date']) - pd.DatetimeIndex(data['booking_date'])).days
        return data

    # create from the arrival data: day of week, month, year, week of year
    def arrival_date_features(self, data):
        '''
        This method creates new variables from the arrival date.
        '''
        data['arrival_dayofweek'] = pd.DatetimeIndex(data['arrival_date']).dayofweek
        data['arrival_month'] = pd.DatetimeIndex(data['arrival_date']).month
        data['arrival_year'] = pd.DatetimeIndex(data['arrival_date']).year
        data['arrival_weekofyear'] = pd.DatetimeIndex(data['arrival_date']).strftime('%W').astype(int)
        data['arrival_date_month'] = data['arrival_date'].dt.strftime('%b')
        #data['arrival_weekofmonth'] = data['arrival_date'].dt.strftime('%V')
        return data
    
    def fit_transform(self, data):
        '''
        This method applies all the feature engineering steps to the data.
        '''
        data = self.is_weekend_stay(data)
        data = self.num_days_stayed(data)
        #data = self.num_weekend_and_week_nights(data)
        data = self.booking_lead_time(data)
        data = self.arrival_date_features(data)
        return data
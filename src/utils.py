import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    @staticmethod
    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

class TrainTestSplitter:  
    '''
    This class contains the methods for splitting the data into train and test sets.
    '''  
    def __init__(self):
        pass

    def split(self,data,codes=[1, 3, 4]): # hotel_codes = [1, 3, 4] resort_codes = [0, 2]
        '''
        This method splits the data into train and test sets.
        '''
        columns_to_keep = ['hotel_name', 'guest_type',
       'customer_type', 'company', 'adr', 'adults', 'children',
       'babies', 'meal', 'market_segment','distribution_channel',
       'assigned_room_type','is_weekend_stay', 'num_days_stayed',
       'booking_lead_time', 'arrival_dayofweek', 'arrival_month',
       'arrival_weekofyear']
        
        #hotel_names = ['Braga City Hotel','Lisbon City Hotel', 'Porto City Hotel']
        #resort_names = ['Algarve Retreat' ,'Duro Valley Resort']

        # make sure the hotel names are encoded, they should be integers
        X = data[data['hotel_name'].isin(codes)][columns_to_keep]
        y = data[data['hotel_name'].isin(codes)]['is_canceled']

        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        return X_train, X_test, y_train, y_test


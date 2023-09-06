import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from tqdm import tqdm


class DataPreprocessor:
    '''
    This class contains the methods for preprocessing the data.
    
    The methods are:
    - convert_datetime
    - remove_cancelled_bookings
    - drop_na_columns
    - drop_duplicates
    - encode_categorical_variables
    
    The fit_transform method applies all the preprocessing steps to the data.
    You can eaiter use the fit_transform method or apply the methods one by one.
    '''
    def __init__(self):
        pass

    def convert_date_columns_to_datetime(self, data):
        '''
        This method converts the date columns to datetime.
        
        The date columns are:
        - arrival_date
        - departure_date
        - booking_date
        '''
        # convert date columns to datetime
        print('Converting date columns to datetime...')
        with tqdm(total=3) as pbar:
            data = data.copy()
            data['arrival_date'] = pd.to_datetime(data['arrival_date'])
            pbar.update(1)
            data['departure_date'] = pd.to_datetime(data['departure_date'])
            pbar.update(1)
            data['booking_date'] = pd.to_datetime(data['booking_date'])
            pbar.update(1)
        print('Date columns converted to datetime.\n')
        return data
    
    def drop_na_rows(self, data):
        '''
        This method drops rows with missing values.
        '''
        # drop rows with missing values
        print('Dropping rows with missing values...')
        number_of_rows_before = data.shape[0]
        with tqdm(total=1) as pbar:
            data = data.copy()
            data.dropna(inplace=True, axis=0)
            pbar.update(1)
        print(f'{abs(data.shape[0]- number_of_rows_before)} rows dropped from {number_of_rows_before}.\n')
        return data

    def drop_duplicates(self, data):
        '''
        This method drops duplicate rows.
        '''
        # drop duplicate rows
        print('Dropping duplicate rows...')
        number_of_rows_before = data.shape[0]
        with tqdm(total=1) as pbar:
            data = data.copy()
            data.drop_duplicates(inplace=True)
            pbar.update(1)
        # print how many rows were dropped
        print(f'{abs(data.shape[0]- number_of_rows_before)} rows dropped from {number_of_rows_before}.\n')
        return data
    
    def drop_negative_adr(self, data):
        '''
        This method drops rows with negative adr.
        '''
        # drop rows with negative adr
        print('Dropping rows with negative adr...')
        number_of_rows_before = data.shape[0]
        with tqdm(total=1) as pbar:
            data = data.copy()
            data = data[data['adr'] >= 0]
            pbar.update(1)
        # print how many rows were dropped
        print(f'{abs(data.shape[0]- number_of_rows_before)} rows dropped from {number_of_rows_before}.\n')
        return data
    
    def scale_numerical_features(self, data, kind='robust'):
        '''
        This method scales the numerical variables.
        The default scaling method is robust scaling.
        '''
        # scale numerical variables
        print('Scaling numerical variables...')
        with tqdm(total=1) as pbar:
            data = data.copy()
            # select numerical columns
            numerical_columns = ['adr','adults', 'children', 'babies','num_days_stayed', 
                                'booking_lead_time', 'arrival_dayofweek', 'arrival_month',
                                'arrival_weekofyear']
            # scale the numerical columns
            if kind == 'robust':
                scaler = RobustScaler()
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
            elif kind == 'standard':
                scaler = StandardScaler()
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
            elif kind == 'minmax':
                scaler = MinMaxScaler()
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
            else:
                print('Please choose a valid scaling method.')
            pbar.update(1)
        print('Numerical variables scaled.\n')
        return data
        

    def encode_categorical_variables(self, data, print_encoding=False):
        '''
        This method encodes the categorical variables using ordinal encoding.
        The encoding is printed for each column.
        The original values are printed for each column.
        '''
        # encode categorical variables
        print('Encoding categorical variables...')
        with tqdm(total=10) as pbar:
            data = data.copy()
            for col in ['hotel_name', 'meal', 'source_country', 'market_segment',
                        'distribution_channel', 'assigned_room_type', 'guest_type',
                        'customer_type', 'season', 'company']:
                # preserve the original values in a new column with the suffix '_original'
                data[col + '_original'] = data[col]
                # use ordinal encoding to encode the categorical variable
                encoder = OrdinalEncoder()
                data[col] = encoder.fit_transform(data[[col]])
                pbar.update(1)
            print('Categorical variables encoded.\n')

        # print the encoding and the original values for each column
        for col in ['hotel_name', 'meal', 'source_country', 'market_segment',
                    'distribution_channel', 'assigned_room_type', 'guest_type',
                    'customer_type', 'season', 'company']:

            encoding_dict = dict(enumerate(encoder.categories_[0]))
            original_values = dict(zip(encoding_dict.values(), data[col + '_original'].unique()))
            if print_encoding:
                print(f'Encoding for {col}:')
                print(f'Original values: {original_values}')
                print(f'Encoded values: {encoding_dict}')
            # remove the original values from the data
            data.drop(columns=[col + '_original'], inplace=True)
        print('\n')
        return data

    def fit_transform(self, data):
        '''
        This method applies all the preprocessing steps to the data.
        '''
        data = self.convert_date_columns_to_datetime(data)
        data = self.drop_na_rows(data)
        data = self.drop_duplicates(data)
        data = self.drop_negative_adr(data)
        data = self.encode_categorical_variables(data)
        return data
    
    def fit_transform_without_encoding_variables(self,data):
        '''
        This method applies all the preprocessing steps to the data.
        Except for the encoding of the categorical variables.
        '''
        data = self.convert_date_columns_to_datetime(data)
        data = self.drop_na_rows(data)
        data = self.drop_duplicates(data)
        data = self.drop_negative_adr(data)
        return data
    
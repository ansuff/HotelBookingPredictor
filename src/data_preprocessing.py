import pandas as pd
from tqdm import tqdm
class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def convert_datetime(self):
        # convert date columns to datetime
        print('Converting date columns to datetime...')
        with tqdm(total=3) as pbar:
            self.data['arrival_date'] = pd.to_datetime(self.data['arrival_date'])
            pbar.update(1)
            self.data['departure_date'] = pd.to_datetime(self.data['departure_date'])
            pbar.update(1)
            self.data['booking_date'] = pd.to_datetime(self.data['booking_date'])
            pbar.update(1)
        print('Date columns converted to datetime.')

    def drop_na_columns(self):
        # drop rows with missing values
        print('Dropping rows with missing values...')
        with tqdm(total=1) as pbar:
            self.data.dropna(inplace=True)
            pbar.update(1)
        print('Rows with missing values dropped.')

    def drop_duplicates(self):
        # drop duplicate rows
        print('Dropping duplicate rows...')
        number_of_rows_before = self.data.shape[0]
        with tqdm(total=1) as pbar:
            self.data.drop_duplicates(inplace=True)
            pbar.update(1)
        # print how many rows were dropped
        print(f'{abs(self.data.shape[0]- number_of_rows_before)} rows dropped.')

    def encode_categorical_variables(self):
        # encode categorical variables
        print('Encoding categorical variables...')
        with tqdm(total=10) as pbar:
            for col in ['hotel_name', 'meal', 'source_country', 'market_segment',
                        'distribution_channel', 'assigned_room_type', 'guest_type',
                        'customer_type', 'season', 'company']:
                # preserve the original values in a new column with the suffix '_original'
                self.data[col + '_original'] = self.data[col]
                self.data[col] = self.data[col].astype('category').cat.codes
                pbar.update(1)
            print('Categorical variables encoded.')

        # print the encoding and the original values for each column
        for col in ['hotel_name', 'meal', 'source_country', 'market_segment',
                    'distribution_channel', 'assigned_room_type', 'guest_type',
                    'customer_type', 'season', 'company']:
            print(f'{col} encoding:')
            encoding_dict = dict(enumerate(self.data[col].astype('category').cat.categories))
            original_values = dict(zip(encoding_dict.values(), self.data[col + '_original'].unique()))
            print(f'Original values: {original_values}')
            # remove the original values from the data
            self.data.drop(columns=[col + '_original'], inplace=True)
            
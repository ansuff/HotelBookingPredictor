import argparse
import os
#import sys

#sys.path.append(os.path.join('src'))

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import HotelBookingFeatures
from src.models import Classification_Model
from src.utils import DataLoader, TrainTestSplitter

NUMERIC_COLUMNS = ['adr','adults', 'children', 'babies','num_days_stayed', 
                                'booking_lead_time', 'arrival_dayofweek', 'arrival_month']


def main(choices):
    # load the data
    data_folder = 'data'
    file_to_open = os.path.join(data_folder, "hotel_bookings.csv")
    data_loader = DataLoader()
    hotel_bookings = data_loader.load_data(file_to_open)

    # preprocess the data
    preprocess = DataPreprocessor()
    hotel_bookings = preprocess.fit_transform(hotel_bookings)

    # extract features
    hotel_booking_features = HotelBookingFeatures()
    hotel_bookings = hotel_booking_features.fit_transform(hotel_bookings)

    # split the data into training and testing sets
    train_test_splitter = TrainTestSplitter()
    X_train_resort, X_test_resort, y_train_resort, y_test_resort = train_test_splitter.split(hotel_bookings, codes=[0, 2])
    X_train_hotel, X_test_hotel, y_train_hotel, y_test_hotel = train_test_splitter.split(hotel_bookings, codes=[1, 3, 4])

    # create and train the hotel model
    print('*'*50)
    print('Proceed to Hotels Models')
    print('*'*50+'\n')
    hotel_model = Classification_Model(X_train_hotel, y_train_hotel, X_test_hotel, y_test_hotel, cols_to_scale=NUMERIC_COLUMNS)
    hotel_model.train_decision_tree()
    hotel_model.train_random_forest()
    hotel_model.train_xgboost()
    hotel_model.train_logistic_regression()
    # evaluate the models
    print('\nEvaluating Hotels Models...\n')
    hotel_model.evaluate()

    if 'confusion_matrix' in choices:
        # print the confusion matrices
        print('\nPrinting Confusion Matrices for Hotels...\n')
        hotel_model.confusion_matrix()

    if 'feature_importance' in choices:
        # print the feature importances
        print('\nPrinting feature importances for Hotels...\n')
        hotel_model.feature_importance()

    # create and train the resort model
    print('*'*50)
    print('Proceed to Resorts Models')
    print('*'*50+'\n')
    resort_model = Classification_Model(X_train_resort, y_train_resort, X_test_resort, y_test_resort, cols_to_scale=NUMERIC_COLUMNS)
    resort_model.train_decision_tree()
    resort_model.train_random_forest()
    resort_model.train_xgboost()
    resort_model.train_logistic_regression()

    # evaluate the models
    print('\nEvaluating Resorts Models...\n')
    resort_model.evaluate()

    if 'feature_importance' in choices:
        # print the feature importances
        print('\nPrinting feature importances for Resorts...\n')
        resort_model.feature_importance()

    if 'confusion_matrix' in choices:
        # print the confusion matrices
        print('\nPrinting Confusion Matrices for Resorts...\n')
        hotel_model.confusion_matrix(plot=False)
        resort_model.confusion_matrix(plot=False)

if __name__ == '__main__':
    # parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('choices', nargs='+', choices=['evaluate', 'confusion_matrix', 'feature_importance'], help='Choose one or more actions to perform')
    args = parser.parse_args()

    # call the main function with the user's choices
    main(args.choices)

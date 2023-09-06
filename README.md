# Hotel Booking Classification

This project is a machine learning model that predicts whether a hotel booking will be canceled or not. The model is trained on a dataset of hotel bookings and uses various classification algorithms to make predictions.

## Installation

To install the required packages, create a virtual environment and run the following command:

```
pip install -r requirements.txt
```

or use poetry:

```
poetry install
```

## Usage

To run the program, use the following command:

```
python main.py [choices]
```

The `choices` parameter is optional and can be one or more of the following options:

- `evaluate`: Evaluate the models and print the accuracy scores.
- `confusion_matrix`: Print the confusion matrices for each model.
- `feature_importance`: Print the feature importances for each model.

If no choices are specified, the program will automatically evaluate the models.

## Data

The dataset used in this project is the [Hotel Booking Demand](https://www.kaggle.com/jessemostipak/hotel-booking-demand) dataset from Kaggle. The dataset contains information about hotel bookings, including the hotel type, booking dates, and customer demographics.

## Models

The program uses four classification algorithms to make predictions:

- Decision Tree
- Random Forest
- XGBoost
- Logistic Regression

The program trains two models: one for resort hotels and one for city hotels. The models are trained on separate datasets and use the same algorithms.

## Results

The program evaluates the models using accuracy scores and confusion matrices. The accuracy scores indicate how well the models can predict whether a booking will be canceled or not. The confusion matrices show the number of true positives, true negatives, false positives, and false negatives for each model.

The program also prints the feature importances for each model. The feature importances indicate which features are most important for making predictions.

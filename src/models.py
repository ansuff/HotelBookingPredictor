import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from tqdm import tqdm
from xgboost import XGBClassifier


class Classification_Model:
    def __init__(self, X_train, y_train, X_test, y_test, cols_to_scale):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cols_to_scale = cols_to_scale

    def train_decision_tree(self):
        print('Training decision tree classifier...')
        with tqdm(total=1) as pbar:
            self.dt = DecisionTreeClassifier()
            self.dt.fit(self.X_train, self.y_train)
            pbar.update(1)
        print('Decision tree classifier trained.\n')

    def train_random_forest(self):
        print('Training random forest classifier...')
        with tqdm(total=1) as pbar:
            self.rf = RandomForestClassifier()
            self.rf.fit(self.X_train, self.y_train)
            pbar.update(1)
        print('Random forest classifier trained.\n')

    def train_logistic_regression(self):
        print('Training logistic regression classifier...')
        with tqdm(total=1) as pbar:
            scaler = RobustScaler()
            X_train_scaled = self.X_train.copy()
            X_train_scaled[self.cols_to_scale] = scaler.fit_transform(X_train_scaled[self.cols_to_scale])
            self.lr = LogisticRegression()
            self.lr.fit(X_train_scaled, self.y_train)
            pbar.update(1)
        print('Logistic regression classifier trained.\n')

    def train_xgboost(self):
        print('Training XGBoost classifier...')
        with tqdm(total=1) as pbar:
            self.xgb = XGBClassifier()
            self.xgb.fit(self.X_train, self.y_train)
            pbar.update(1)
        print('XGBoost classifier trained.\n')

    def confusion_matrix(self):
        print('Creating confusion matrix...')
        models = {'Decision Tree': self.dt, 'Random Forest': self.rf, 'Logistic Regression': self.lr, 'XGBoost': self.xgb}
        with tqdm(total=len(models)) as pbar:
            for name, model in models.items():
                if name == 'Logistic Regression':
                    scaler = RobustScaler()
                    X_test_scaled = self.X_test.copy()
                    X_test_scaled[self.cols_to_scale] = scaler.fit_transform(X_test_scaled[self.cols_to_scale])
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)
                print(f'Confusion matrix for {name}:')
                print(cm)
                pbar.update(1)
        print('Confusion matrices created.\n')
    
    def plot_feature_importance(self):
        print('Plotting feature importance...')
        models = {'Decision Tree': self.dt, 'Random Forest': self.rf, 'XGBoost': self.xgb}
        feature_importances = pd.DataFrame(columns=['feature', 'importance', 'model'])
        with tqdm(total=len(models)) as pbar:
            for name, model in models.items():
                feature_importances_model = pd.DataFrame({'feature': self.X_train.columns, 'importance': model.feature_importances_})
                feature_importances_model['model'] = name
                feature_importances = pd.concat([feature_importances, feature_importances_model])
                pbar.update(1)
        feature_importances = feature_importances.groupby(['feature', 'model']).mean().reset_index()
        # for each model sort by highest importance
        for name, model in models.items():
            feature_importances_model = feature_importances[feature_importances['model'] == name].sort_values(by='importance', ascending=False)
            print(f'Feature importance for {name}:')
            print(feature_importances_model)
        #print(feature_importances)
        print('Feature importance plotted.\n')
        return feature_importances
    
    def evaluate(self,cv=5):
        print('Evaluating classifiers with cross validation...')
        models = {'Decision Tree': self.dt, 'Random Forest': self.rf, 'Logistic Regression': self.lr, 'XGBoost': self.xgb}
        results = []
        with tqdm(total=len(models)) as pbar:
            for name, model in models.items():
                if name == 'Logistic Regression':
                    scaler = RobustScaler()
                    X_test_scaled = self.X_test.copy()
                    X_test_scaled[self.cols_to_scale] = scaler.fit_transform(X_test_scaled[self.cols_to_scale])
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                report = classification_report(self.y_test, y_pred)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                results.append([name, accuracy, precision, recall, f1, report, cv_mean, cv_std])
                pbar.update(1)
        headers = ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Classification Report', 'CV Mean', 'CV Std']
        print(tabulate(results, headers=headers, tablefmt='orgtbl'))

    def predict(self, data):
        print('Predicting on new data...')
        models = {'Decision Tree': self.dt, 'Random Forest': self.rf, 'Logistic Regression': self.lr, 'XGBoost': self.xgb}
        with tqdm(total=len(models)) as pbar:
            for name, model in models.items():
                if name == 'Logistic Regression':
                    scaler = RobustScaler()
                    data_scaled = data.copy()
                    data_scaled[self.cols_to_scale] = scaler.fit_transform(data_scaled[self.cols_to_scale])
                    y_pred = model.predict(data_scaled)
                else:
                    y_pred = model.predict(data)
                print(f'Predictions for {name}:')
                print(y_pred)
                pbar.update(1)
        print('Predictions made.\n')

class forecasting_model:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_ARIMA(self):
        print('Training ARIMA model...')
        # find the best order for ARIMA model
        self.arima = auto_arima(self.y_train, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
        # fit the ARIMA model
        self.arima.fit(self.y_train)
        print('ARIMA model trained.\n')

    def train_SARIMA(self):
        print('Training SARIMA model...')
        # find the best order for SARIMA model
        self.sarima = auto_arima(self.y_train, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
        # fit the SARIMA model
        self.sarima.fit(self.y_train)
        print('SARIMA model trained.\n')

    def evaluate(self):
        print('Evaluating models...')
        models = {'ARIMA': self.arima, 'SARIMA': self.sarima}
        results = []
        with tqdm(total=len(models)) as pbar:
            for name, model in models.items():
                y_pred = model.predict(len(self.y_test))
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                mae = mean_absolute_error(self.y_test, y_pred)
                results.append([name, rmse, mae])
                pbar.update(1)
        headers = ['Model', 'RMSE', 'MAE']
        print(tabulate(results, headers=headers, tablefmt='orgtbl'))


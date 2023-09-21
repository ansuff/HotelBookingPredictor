import unittest
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import Classification_Model


class TestClassificationModel(unittest.TestCase):
    def setUp(self):
        # create sample data for testing
        self.X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        self.y_train = pd.Series([0, 1, 0])
        self.X_test = pd.DataFrame({'feature1': [4, 5, 6], 'feature2': [7, 8, 9]})
        self.y_test = pd.Series([1, 0, 1])
        self.cols_to_scale = ['feature1']
        self.model = Classification_Model(self.X_train, self.y_train, self.X_test, self.y_test, self.cols_to_scale)

    def test_train_decision_tree(self):
        self.model.train_decision_tree()
        self.assertIsNotNone(self.model.dt)

    def test_train_random_forest(self):
        self.model.train_random_forest()
        self.assertIsNotNone(self.model.rf)

    def test_train_logistic_regression(self):
        self.model.train_logistic_regression()
        self.assertIsNotNone(self.model.lr)

    def test_train_xgboost(self):
        self.model.train_xgboost()
        self.assertIsNotNone(self.model.xgb)

    def test_confusion_matrix(self):
        self.model.train_decision_tree()
        self.model.train_random_forest()
        self.model.train_logistic_regression()
        self.model.train_xgboost()
        self.model.confusion_matrix(plot=False)

    def test_feature_importance(self):
        self.model.train_decision_tree()
        self.model.train_random_forest()
        self.model.train_xgboost()
        feature_importances = self.model.feature_importance(plot=False)
        self.assertIsNotNone(feature_importances)

    def test_evaluate(self):
        self.model.train_decision_tree()
        self.model.train_random_forest()
        self.model.train_logistic_regression()
        self.model.train_xgboost()
        self.model.evaluate(cv=2)

    def test_predict(self):
        self.model.train_decision_tree()
        self.model.train_random_forest()
        self.model.train_logistic_regression()
        self.model.train_xgboost()
        data = pd.DataFrame({'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})
        self.model.predict(data)

if __name__ == '__main__':
    unittest.main()
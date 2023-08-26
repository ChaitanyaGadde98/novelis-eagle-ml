import os
import numpy as np
import pandas as pd
import joblib

from config import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model


class EnsembleModelBuilderAndEvaluator:

    def __init__(self, data, rf_model_path=RF_MODEL, xgb_model_path=XGB_MODEL, lstm_model_path=LSTM_MODEL, mode='train'):
        self.data = data
        self.X = self.data.drop(columns=['Good/Bad'])
        self.y = self.data["Good/Bad"].astype(int)

        if mode == 'train':
            self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.xgb_clf = XGBClassifier(random_state=42)
        elif mode == 'test':
            if rf_model_path and xgb_model_path and lstm_model_path:
                self.rf_clf = joblib.load(RF_MODEL)
                self.xgb_clf = joblib.load(XGB_MODEL)
                self.lstm_model = load_model(LSTM_MODEL)
                self.merged_data = pd.read_csv("data/test/merged_unprocessed.csv")
            else:
                raise ValueError("Please provide valid model paths for testing.")
        else:
            raise ValueError("Invalid mode. Choose 'train' or 'test'.")
        self.mode = mode

    def _ensemble_predict(self, X):
        rf_pred = self.rf_clf.predict(X).astype(int)
        xgb_pred = self.xgb_clf.predict(X).astype(int)
        X_reshaped = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
        lstm_pred = (self.lstm_model.predict(X_reshaped) > 0.5).astype(int).squeeze()

        print(np.sum(rf_pred == 0))
        print(np.sum(rf_pred == 1))
        print(np.sum(xgb_pred == 0))
        print(np.sum(xgb_pred == 1))
        print(np.sum(lstm_pred == 0))
        print(np.sum(lstm_pred == 1))


        combined_predictions = np.vstack((rf_pred, xgb_pred, lstm_pred))
        
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=combined_predictions)
        return majority_vote


    def train_and_evaluate(self):
        if self.mode != 'train':
            raise ValueError("Cannot train in 'test' mode.")

        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        precisions, recalls, accuracies, f1_scores = [], [], [], []

        for i, (train_index, test_index) in enumerate(stratified_kfold.split(self.X, self.y)):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # Train the ensemble models
            self.rf_clf.fit(X_train, y_train)
            self.xgb_clf.fit(X_train, y_train)

            # You need to add LSTM training here.

            # Ensemble prediction
            y_pred = self._ensemble_predict(X_test)
            precision, recall, f1, _ = classification_report(y_test, y_pred, output_dict=True)['1'].values()
            accuracy = (y_pred == y_test).sum() / len(y_test)

            precisions.append(precision)
            recalls.append(recall)
            accuracies.append(accuracy)
            f1_scores.append(f1)

        return {
            "precisions": precisions,
            "recalls": recalls,
            "accuracies": accuracies,
            "f1_scores": f1_scores
        }

    def test(self, return_predictions=False):
        if self.mode != 'test':
            raise ValueError("Cannot test in 'train' mode.")
        
        y_pred = self._ensemble_predict(self.X)
        
        # Saving individual features and their predictions
        feature_columns = self.X.columns.tolist()
        features_df = pd.DataFrame(self.X.values, columns=feature_columns)
        features_df['Good/Bad'] = self.y.values
        features_df['Predicted'] = y_pred
        features_df['Prediction_Status'] = [' ' if a == p else 'X' for a, p in zip(self.y, y_pred)]
        features_df.to_csv('data/test/test_results.csv', index=True)
        
        # Merging with unprocessed data
        self.merged_data['Good/Bad'] = self.y.values
        self.merged_data['Predicted'] = y_pred
        self.merged_data['Prediction_Status'] = [' ' if a == p else 'X' for a, p in zip(self.y, y_pred)]
        self.merged_data.to_csv('data/test/merged_test_results.csv', index=True)
        
        if return_predictions:
            return y_pred
        else:
            precision, recall, f1, _ = classification_report(self.y, y_pred, output_dict=True)['1'].values()
            accuracy = (y_pred == self.y).sum() / len(self.y)
            cm = confusion_matrix(self.y, y_pred)

            return {
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": cm.tolist()
            }


    def save_models(self, rf_filename, xgb_filename, lstm_filename):
        if self.mode != 'train':
            raise ValueError("Cannot save models in 'test' mode.")

        joblib.dump(self.rf_clf, rf_filename)
        joblib.dump(self.xgb_clf, xgb_filename)
        self.lstm_model.save(lstm_filename)
        return f"Models saved to {rf_filename}, {xgb_filename}, and {lstm_filename}"



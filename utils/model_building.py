import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from config import FIGURES_PATH
import pandas as pd


class ModelBuilderAndEvaluator:

    def __init__(self, data, mode='train', model_path=None):
        self.data = data
        self.X = self.data.drop(columns=['Good/Bad'])
        self.y = self.data["Good/Bad"].astype(int)

        # Check mode (train or test)
        if mode == 'train':
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        elif mode == 'test':
            if model_path:
                self.clf = joblib.load(model_path)
                self.merged_data = pd.read_csv("data/test/merged_unprocessed.csv")
            else:
                raise ValueError("Please provide a valid model path for testing.")
        else:
            raise ValueError("Invalid mode. Choose 'train' or 'test'.")

        self.figures_path = os.path.join(os.getcwd(), FIGURES_PATH)
        self.mode = mode

    def train_and_evaluate(self):
        if self.mode != 'train':
            raise ValueError("Cannot train in 'test' mode.")
        precisions, recalls, accuracies, f1_scores = self._train_evaluate_with_cross_validation()
        self._plot_metrics(precisions, recalls, accuracies, f1_scores)

        from sklearn import tree

        desc_features = self.data.columns.to_list()[:-1]

        # plot = tree.plot_tree(self.clf, feature_names=desc_features, class_names=["Good", "Bad"], proportion='True')
        feature_importances = dict(zip(desc_features, self.clf.feature_importances_))
        sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

        print("Features:", sorted_feature_importances)
        return {
            "precisions": precisions,
            "recalls": recalls,
            "accuracies": accuracies,
            "f1_scores": f1_scores
        }

    def _train_evaluate_with_cross_validation(self):
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        precisions = []
        recalls = []
        accuracies = []
        f1_scores = []

        for i, (train_index, test_index) in enumerate(stratified_kfold.split(self.X, self.y)):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)
            precision, recall, f1, _ = classification_report(y_test, y_pred, output_dict=True)['1'].values()
            accuracy = (y_pred == y_test).sum() / len(y_test)

            precisions.append(precision)
            recalls.append(recall)
            accuracies.append(accuracy)
            f1_scores.append(f1)

            self._save_confusion_matrix_plot(y_test, y_pred, i)

        return precisions, recalls, accuracies, f1_scores

    def _plot_metrics(self, precisions, recalls, accuracies, f1_scores):
        metrics = ['Accuracy', 'Recall', 'F1 Score', 'Precision']
        values = [accuracies, recalls, f1_scores, precisions]

        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics):
            plt.plot(range(1, 6), values[i], marker='o', label=metric)
        plt.title('Metrics for Each Stratum')
        plt.xlabel('Stratum')
        plt.ylabel('Value')
        plt.xticks(range(1, 6))
        plt.legend()
        plt.grid(True)
        metrics_filename = os.path.join(self.figures_path, 'train/metrics_by_stratum.png')
        plt.savefig(metrics_filename)
        plt.close()

    def _save_confusion_matrix_plot(self, y_true, y_pred, fold_index):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - Fold {fold_index + 1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_filename = os.path.join(self.figures_path, f'train/confusion_matrix_fold_{fold_index + 1}.png')
        plt.savefig(cm_filename)
        plt.close()

    def test(self, return_predictions=False):
        if self.mode != 'test':
            raise ValueError("Cannot test in 'train' mode.")

        y_pred = self.clf.predict(self.X)
        feature_columns = self.X.columns.tolist()

        features_df = pd.DataFrame(self.X.values, columns=feature_columns)

        features_df['Good/Bad'] = self.y.values
        features_df['Predicted'] = y_pred

        features_df['Prediction_Status'] = [' ' if a == p else 'X' for a, p in zip(self.y, y_pred)]
        features_df.to_csv('data/test/test_results.csv', index=True)

        self.merged_data['Good/Bad'] = self.y.values
        self.merged_data['Predicted'] = y_pred
        self.merged_data['Prediction_Status'] = [' ' if a == p else 'X' for a, p in zip(self.y, y_pred)]
        self.merged_data.to_csv('data/test/merged_test_results.csv', index=True)

        if return_predictions:
            return y_pred
        else:
            precision, recall, f1, _ = classification_report(self.y, y_pred, output_dict=True)['1'].values()
            accuracy = (y_pred == self.y).sum() / len(self.y)

            # Save confusion matrix for test
            cm = confusion_matrix(self.y, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title('Confusion Matrix - Test')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            cm_filename = os.path.join(self.figures_path, 'test/confusion_matrix_test.png')
            plt.savefig(cm_filename)
            plt.close()

            # Save bar plots for scores
            scores = [accuracy, recall, f1, precision]
            labels = ['Accuracy', 'Recall', 'F1 Score', 'Precision']
            plt.figure(figsize=(10, 6))
            plt.bar(labels, scores, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
            plt.title('Evaluation Metrics for Test Data')
            plt.ylim(0, 1)
            plt.ylabel('Score')
            score_filename = os.path.join(self.figures_path, 'test/scores_test.png')
            plt.savefig(score_filename)
            plt.close()

            return {
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": cm.tolist()
            }

    def save_model(self, filename):
        if self.mode != 'train':
            raise ValueError("Cannot save model in 'test' mode.")
        joblib.dump(self.clf, filename)
        return f"Model saved to {filename}"

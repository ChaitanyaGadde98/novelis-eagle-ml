import numpy as np
from config import COLS_TO_DROP
from sklearn.neighbors import NearestNeighbors
import pandas as pd


class DataReaderAndMerger:
    def __init__(self, sensor_csv, high_freq_csv, percent_reference_csv, trainTest="train"):
        self.sensor_csv = sensor_csv
        self.high_freq_csv = high_freq_csv
        self.percent_reference_csv = percent_reference_csv
        self.trainTest = trainTest

    def read_and_merge(self):
        sensor_data = pd.read_csv(self.sensor_csv)
        sensor_high_freq_data = pd.read_csv(self.high_freq_csv)
        percent_reference_data = pd.read_csv(self.percent_reference_csv)

        # sensor_high_freq_data['Percent'] = pd.to_numeric(sensor_high_freq_data['Percent'], errors='coerce')
        # sensor_high_freq_data.dropna(subset=['Percent'], inplace=True)
        #
        # merged_df = pd.merge_asof(sensor_high_freq_data.sort_values('Percent'),
        #                           percent_reference_data.sort_values('Percent Min'),
        #                           left_on='Percent',
        #                           right_on='Percent Min',
        #                           direction='forward')
        #
        # final_df = pd.merge(sensor_data, merged_df, on='timestamp', how='left')

        sensor_merged = pd.merge(sensor_data, sensor_high_freq_data, on='timestamp', how='left')

        sensor_merged['Percent'] = pd.to_numeric(sensor_merged['Percent'], errors='coerce')

        percent_reference_merged = []

        for _, row in sensor_merged.iterrows():
            matching_row = percent_reference_data[
                (row['Percent'] > percent_reference_data['Percent Min']) &
                (row['Percent'] < percent_reference_data['Percent Max'])
                ]
            if not matching_row.empty:
                percent_reference_merged.append(matching_row.iloc[0])
            else:
                percent_reference_merged.append(pd.Series())

        percent_reference_merged_df = pd.DataFrame(percent_reference_merged)
        sensor_merged.reset_index(drop=True, inplace=True)
        percent_reference_merged_df.reset_index(drop=True, inplace=True)

        final_merged = pd.concat([sensor_merged, percent_reference_merged_df], axis=1)

        target = 'Good/Bad'
        order_cols = [col for col in final_merged.columns if col != target] + [target]
        final_df = final_merged[order_cols]

        if self.trainTest == "train":
            final_df.to_csv("data/train_merged.csv")
        else:
            final_df.to_csv("data/test_merged.csv")
        return final_df


class DataPreprocessing:

    def __init__(self, data):
        self.data = data.copy()

    def meanImputation(self):
        self.data.fillna(self.data.mean(), inplace=True)

    def medianImputation(self):
        self.data.fillna(self.data.median(), inplace=True)

    def dropRows(self):
        essential_cols = ['Period Code', 'Cycle ID', 'Good/Bad']
        self.data.dropna(subset=essential_cols, inplace=True)

    def typCasteFeatures(self):
        non_numeric_cols = self.data.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data.fillna(self.data.median(), inplace=True)

    def preprocessTargets(self):
        self.data = self.data[self.data['Good/Bad'].isin(['0', '1'])]
        # self.data['Good/Bad'] = self.data['Good/Bad'].replace('Bad', '1').astype(int)

    def outliersSTD(self, threshold=3.0):
        target_column = 'Good/Bad'
        for col in [c for c in self.data.columns if c != target_column]:
            median = self.data[col].median()
            mean = self.data[col].mean()
            std = self.data[col].std()
            is_outlier = (self.data[col] < (mean - threshold * std)) | (self.data[col] > (mean + threshold * std))
            self.data.loc[is_outlier, col] = median

    def outliersIQR(self):
        target_column = 'Good/Bad'
        for col in [c for c in self.data.columns if c != target_column]:  # We skip the target column
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            self.data.loc[(self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR)), col] = None
            self.data[col].fillna(self.data[col].median(), inplace=True)

    def dropColumns(self):
        self.data.drop(columns=COLS_TO_DROP, inplace=True)

    def normalizeData(self):
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())

    def apply_SMOTE(self, k=5):
        features = self.data.drop(columns=['Good/Bad']).to_numpy()
        target = self.data['Good/Bad'].to_numpy()

        def SMOTE(samples, n_samples, k=k):
            nn = NearestNeighbors(n_neighbors=k + 1).fit(samples)
            _, neighbors = nn.kneighbors(samples)
            neighbors = neighbors[:, 1:]

            synthetic_samples = []
            for i in range(n_samples):
                sample_idx = np.random.randint(0, len(samples))
                sample = samples[sample_idx]
                neighbor_idx = np.random.choice(neighbors[sample_idx])
                neighbor = samples[neighbor_idx]
                gap = np.random.random()
                synthetic_sample = sample + gap * (neighbor - sample)
                synthetic_samples.append(synthetic_sample)

            return np.array(synthetic_samples)

        n_samples_needed = np.bincount(target)[0] - np.bincount(target)[1]

        minority_samples = features[target == 1]
        synthetic_samples = SMOTE(minority_samples, n_samples_needed)

        synthetic_data = np.column_stack((synthetic_samples, np.ones(n_samples_needed, dtype=int)))
        augmented_data = np.vstack((self.data.to_numpy(), synthetic_data))

        self.data = pd.DataFrame(augmented_data, columns=self.data.columns)
        self.data['Good/Bad'] = self.data['Good/Bad'].astype(int)

    def getProcessedData(self):
        return self.data

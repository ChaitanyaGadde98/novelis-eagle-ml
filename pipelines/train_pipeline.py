from utils.preprocessor import DataPreprocessing, DataReaderAndMerger
from utils.model_building import ModelBuilderAndEvaluator
from config import *

# data reading and mergering
data_reader_merger = DataReaderAndMerger(TRAIN_SENSOR_PATH, TRAIN_HIGH_FREQ_PATH, TRAIN_PERCENT_REF_PATH, trainTest="train")
dataset = data_reader_merger.read_and_merge()

# data preprocessing
data_preprocessing = DataPreprocessing(dataset)
data_preprocessing.medianImputation()
data_preprocessing.dropRows()
data_preprocessing.preprocessTargets()
data_preprocessing.typCasteFeatures()
# data_preprocessing.outliersIQR()
data_preprocessing.outliersSTD()
data_preprocessing.dropColumns()
# data_preprocessing.normalizeData()
data_preprocessing.apply_SMOTE()
preprocessed_data = data_preprocessing.getProcessedData()
preprocessed_data.to_csv("data/processed_merged_data.csv")


# model building
model_builder = ModelBuilderAndEvaluator(preprocessed_data)
cross_val_scores = model_builder.train_and_evaluate()
save_message = model_builder.save_model(MODEL_PATH)

print(f"Cross Validation Scores: {cross_val_scores}")
print(save_message)

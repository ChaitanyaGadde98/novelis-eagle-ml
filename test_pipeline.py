# from utils.preprocessor import DataPreprocessing, DataReaderAndMerger
# from utils.model_building import ModelBuilderAndEvaluator
# from config import *
#
# data_reader_merger = DataReaderAndMerger(TEST_SENSOR_PATH, TEST_HIGH_FREQ_PATH, TEST_PERCENT_REF_PATH, trainTest="test")
# dataset = data_reader_merger.read_and_merge()
# dataset.to_csv("data/test/merged_unprocessed.csv")
# # data preprocessing
# data_preprocessing = DataPreprocessing(dataset)
# data_preprocessing.medianImputation()
# # data_preprocessing.dropRows()
# # data_preprocessing.preprocessTargets()
# data_preprocessing.typCasteFeatures()
# # data_preprocessing.outliersIQR()
# # data_preprocessing.outliersSTD()
# data_preprocessing.dropColumns()
# preprocessed_data = data_preprocessing.getProcessedData()
#
# # model evaluation
# model_evaluator = ModelBuilderAndEvaluator(preprocessed_data, mode='test', model_path=MODEL_PATH)
# results = model_evaluator.test(return_predictions=False)
#
# if isinstance(results, dict):
#     precision = results.get("precision")
#     recall = results.get("recall")
#     accuracy = results.get("accuracy")
#     f1_score = results.get("f1_score")
#     print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, F1 Score: {f1_score}")
# else:  # It means predictions were returned
#     print("Predictions:", results)



from utils.preprocessor import DataPreprocessing, DataReaderAndMerger
from utils.ensemble_model import EnsembleModelBuilderAndEvaluator
from config import *

data_reader_merger = DataReaderAndMerger(TEST_SENSOR_PATH, TEST_HIGH_FREQ_PATH, TEST_PERCENT_REF_PATH, trainTest="test")
dataset = data_reader_merger.read_and_merge()
dataset.to_csv("data/test/merged_unprocessed.csv")
# data preprocessing
data_preprocessing = DataPreprocessing(dataset)
data_preprocessing.medianImputation()
# data_preprocessing.dropRows()
# data_preprocessing.preprocessTargets()
data_preprocessing.typCasteFeatures()
# data_preprocessing.outliersIQR()
# data_preprocessing.outliersSTD()
data_preprocessing.dropColumns()
preprocessed_data = data_preprocessing.getProcessedData()

# model evaluation

ensemble_evaluator = EnsembleModelBuilderAndEvaluator(preprocessed_data, mode='test')
results = ensemble_evaluator.test(return_predictions=False)

if isinstance(results, dict):
    precision = results.get("precision")
    recall = results.get("recall")
    accuracy = results.get("accuracy")
    f1_score = results.get("f1_score")
    print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, F1 Score: {f1_score}")
else:  # It means predictions were returned
    print("Predictions:", results)
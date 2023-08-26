import pandas as pd
from flask import Flask, request, jsonify, render_template
from utils.preprocessor import DataPreprocessing, DataReaderAndMerger
from utils.model_building import ModelBuilderAndEvaluator
from utils.ensemble_model import EnsembleModelBuilderAndEvaluator
import os
from config import *

app = Flask(__name__)


# redirect homepage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    keywords_list = ['Percent_reference', 'Sensor_high_freq', 'Sensor']
    files = request.files.getlist('files[]')

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    detected_keywords = []

    for file in files:
        if file.filename.endswith('.csv'):
            for keyword in keywords_list:
                # Check if keyword is 'Sensor' and filename contains 'Sensor' but not 'high'
                if keyword == "Sensor" and "Sensor" in file.filename and "high" not in file.filename:
                    file_path = os.path.join(DATA_FOLDER, f"{keyword}.csv")
                    file.save(file_path)
                    detected_keywords.append(keyword)
                    break
                # Check for other keywords
                elif keyword in file.filename:
                    file_path = os.path.join(DATA_FOLDER, f"{keyword}.csv")
                    file.save(file_path)
                    detected_keywords.append(keyword)
                    break

    if all(keyword in detected_keywords for keyword in keywords_list):
        return jsonify({'success': True, 'message': 'Files successfully uploaded.'}), 200

    return jsonify({'success': False, 'message': 'Required files not uploaded'}), 400


@app.route('/predict_single', methods=['GET'])
def predict_single():
    print("predict")
    data_reader_merger = DataReaderAndMerger(TEST_SENSOR_PATH, TEST_HIGH_FREQ_PATH, TEST_PERCENT_REF_PATH,
                                             trainTest="test")
    dataset = data_reader_merger.read_and_merge()
    print("[INFO] Dataset Merger - Success")
    # data preprocessing
    data_preprocessing = DataPreprocessing(dataset)
    data_preprocessing.medianImputation()
    print("[INFO] Median Imputation - Success")
    # data_preprocessing.dropRows()
    # print("[INFO] Drop Rows - Success")
    # data_preprocessing.preprocessTargets()
    # print("[INFO] Targets Preprocess - Success")
    data_preprocessing.typCasteFeatures()
    print("[INFO] Type Casting - Success")
    # data_preprocessing.outliersIQR()
    # data_preprocessing.outliersSTD()
    # print("[INFO] OutliersSTD - Success")
    data_preprocessing.dropColumns()
    print("[INFO] Drop Columns - Success")
    preprocessed_data = data_preprocessing.getProcessedData()

    # model evaluation
    model_evaluator = ModelBuilderAndEvaluator(preprocessed_data, mode='test', model_path=MODEL_PATH)
    scores = model_evaluator.test(return_predictions=False)
    # with open("temp/test_results.csv", 'r') as file:
    #     csv_data = file.read()
    print(scores)
    return jsonify({"scores": scores}), 200

@app.route('/predict', methods=['GET'])
def predict():
    print("predict")
    data_reader_merger = DataReaderAndMerger(TEST_SENSOR_PATH, TEST_HIGH_FREQ_PATH, TEST_PERCENT_REF_PATH,
                                             trainTest="test")
    dataset = data_reader_merger.read_and_merge()
    print("[INFO] Dataset Merger - Success")
    # data preprocessing
    data_preprocessing = DataPreprocessing(dataset)
    data_preprocessing.medianImputation()
    print("[INFO] Median Imputation - Success")
    # data_preprocessing.dropRows()
    # print("[INFO] Drop Rows - Success")
    # data_preprocessing.preprocessTargets()
    # print("[INFO] Targets Preprocess - Success")
    data_preprocessing.typCasteFeatures()
    print("[INFO] Type Casting - Success")
    # data_preprocessing.outliersIQR()
    # data_preprocessing.outliersSTD()
    # print("[INFO] OutliersSTD - Success")
    data_preprocessing.dropColumns()
    print("[INFO] Drop Columns - Success")
    preprocessed_data = data_preprocessing.getProcessedData()

    # model evaluation
    ensemble_evaluator = EnsembleModelBuilderAndEvaluator(preprocessed_data, mode='test')
    scores = ensemble_evaluator.test(return_predictions=False)
    # with open("temp/test_results.csv", 'r') as file:
    #     csv_data = file.read()
    print(scores)
    return jsonify({"scores": scores}), 200


@app.route('/download_preds', methods=['GET'])
def download_preds():
    with open("data/test/merged_test_results.csv", 'r') as file:
        csv_data = file.read()
    return csv_data, 200, {'Content-Type': 'text/csv'}


if __name__ == '__main__':
    app.run(debug=True, port=PORT)

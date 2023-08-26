SENSOR_PATH = "data/Sensor.csv"
HIGH_FREQ_PATH = "data/Sensor_high_freq.csv"
PERCENT_REF_PATH = "data/Percent_reference.csv"

TEST_SENSOR_PATH = "data/test/Sensor.csv"
TEST_HIGH_FREQ_PATH = "data/test/Sensor_high_freq.csv"
TEST_PERCENT_REF_PATH = "data/test/Percent_reference.csv"

TRAIN_SENSOR_PATH = "data/train/Sensor.csv"
TRAIN_HIGH_FREQ_PATH = "data/train/Sensor_high_freq.csv"
TRAIN_PERCENT_REF_PATH = "data/train/Percent_reference.csv"

BEST_MODEL = "model/random_forest_model_best.pkl"

MODEL_PATH = "model/random_forest_model.pkl"
FIGURES_PATH = "figures"
DATA_FOLDER = "data/test"
REQUIRED_FILES = ["Sensor", "Sensor_high_freq", "Percent_reference"]
PORT = 5000
COLS_TO_DROP = ['Period Code', 'Cycle ID', 'B_4', 'B_5', 'B_9','B_10', 'B_14','B_20', 'B_22', 'B_23', 'timestamp']

RF_MODEL = "model/random_forest_model.pkl"
XGB_MODEL = "model/xgb_best.pkl"
LSTM_MODEL = "model/lstm_best_model.h5"
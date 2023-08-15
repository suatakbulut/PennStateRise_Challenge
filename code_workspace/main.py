import argparse
import warnings

from data_prep import prepare_train_test_split
from utils_data import *
from utils_model import *

warnings.filterwarnings("ignore")

# Utilize Command Line Arguments when deciding on new data creation and test data size
parser = argparse.ArgumentParser(description='Details')
parser.add_argument('--create_data', '-cd', dest='create_data',
                    type=int, help='1 for recreating the dataset')
parser.add_argument('--test_size',   '-ts', dest='test_size',
                    type=float, help='test_size for train test split')
args = parser.parse_args()

if not args.create_data:
    print("Data re-computation is not specified.. Checking if the data already exists..")
    use_saved_one = True
else:
    use_saved_one = 1 - args.create_data


if not args.test_size:
    print("Test_size is not specified.. Employing 0.40")
    test_size = 0.40
else:
    test_size = args.test_size

if __name__ == "__main__":
    # unwanted_column_endings = ["_std", "race"]
    unwanted_column_endings = []

    X_train, X_test, y_train, y_test = prepare_train_test_split(
        use_saved_one=use_saved_one,
        unwanted_column_endings=unwanted_column_endings,
        test_size=test_size,
    )
    # Check a few simple model results
    for predictor in ("logistic", "forest", "xgb"):
        single_model_results(predictor, X_train, X_test, y_train, y_test)

    # Run multiple gridsearchcv models and save them
    results_df = train_multiple_gridsearch(X_train, X_test, y_train, y_test)

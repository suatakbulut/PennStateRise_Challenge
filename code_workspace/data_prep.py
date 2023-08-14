from sklearn.model_selection import train_test_split

from utils_data import *
from utils_model import categorize_columns, get_preprocessor, select_columns


def get_data(use_saved_one=True):
    print("\nDATA PREP")
    print("=========\n")
    processed_data_path = get_processed_data_path()
    processed_file_path = processed_data_path.split("/")[-1]

    if use_saved_one:
        if os.path.exists(processed_data_path):
            print(f"Loading processed data from {processed_data_path}..")
            df = pd.read_pickle(processed_data_path)
        else:
            print("Processed data does not exist. Creating one.")
            df = prepare_data()
            print(
                f"Processed data is created and saved at {processed_data_path}..")
            df.to_pickle(processed_data_path)
    else:
        df = prepare_data()
        print(
            f"Processed data is created and saved at {processed_data_path}..")
        df.to_pickle(processed_data_path)

    return df


def prepare_train_test_split(use_saved_one=True, unwanted_column_endings=[], test_size=0.30):
    df = get_data(use_saved_one=use_saved_one)
    selected_columns = select_columns(df, unwanted_column_endings=unwanted_column_endings)

    X = df[selected_columns]
    y = df["Outcome 30 days Hospitalization"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"Shapes: X_train: {X_train.shape} - X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test

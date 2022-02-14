'''
This file contains the tests and logging associated with the churn library script

Author: Ku Wei Chiet
Date: 21-Dec-2021
'''
import os
import logging
import churn_library as cls

# Initiate logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    Test 'import_data' function
    '''

    try:
        df = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    # Checking if the size of data frame is zero or not
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_eda(perform_eda, df):
    '''
    Test 'perform_eda' function
    '''

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(df)
    path = "./images/eda"

    # Checking if the list is empty or not
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: It does not appear that the you "
                        "are correctly saving images to the eda folder.")
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    Test 'encoder_helper' function
    '''

    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    df = encoder_helper(df, cat_columns, 'Churn')

    try:
        for col in cat_columns:
            assert col in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe appears to be missing the "
            "transformed categorical columns")
        return err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    Test 'perform_feature_engineering' function
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    # Check the size of training and testing data set
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "The four objects that should be returned were not.")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    Test 'train_models' function
    '''

    train_models(X_train, X_test, y_train, y_test)
    path = "./images/results/"

    # Check whether the plots are saved
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: Results image files not found")
        raise err

    # Check whether the models are saved
    path = "./models/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Model files not found")
        raise err


if __name__ == "__main__":
    DATA_FRAME = test_import(cls.import_data)
    test_eda(cls.perform_eda, DATA_FRAME)
    DATA_FRAME = test_encoder_helper(cls.encoder_helper, DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA_FRAME)
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

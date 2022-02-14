'''
This is a library of functions used for the Udacity project aimed at finding
customers who are likely to churn.

Author: Ku Wei Chiet
Date: 21-Dec-2021
'''
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(path):
    '''
    Read and returns the dataframe for the csv file found at 'path'

    Input:
            path: a path to the csv to be read
    Output:
            df: pandas dataframe
    '''
    df = pd.read_csv(path)

    return df


def perform_eda(df):
    '''
    Perform EDA on df and save figures to images folder.
    The images will be save in folder './images/eda/' of current path.

    Input:
            df: pandas dataframe

    Output:
            None
    '''
    # Plot churn histogram distribution
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn Status");
    plt.ylabel("Number of Customers");
    plt.savefig('./images/eda/churn_distribution.png')
    plt.close()

    # Plot customer age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title("Customer Age Distribution")
    plt.xlabel("Age");
    plt.ylabel("Number of Customers");
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.close()

    # Plot marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Customer Marital Status Distribution")
    plt.xlabel("Marital Status");
    plt.ylabel("Percentages of Customers");
    plt.savefig('./images/eda/marital_status_distribution.png')
    plt.close()

    # Plot total transaction ct distribution
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.title("Total Transaction Ct Distribution")
    plt.xlabel("Total Transaction ct ");
    plt.ylabel("Density");
    plt.savefig('./images/eda/total_transaction_distribution.png')
    plt.close()

    # Plot churn and categorical features correlation heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Churn and Categorical Features Correlation Heatmap");
    plt.savefig('./images/eda/heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    Output:
            df: pandas dataframe with new columns for
    '''

    for col in category_lst:
        new_lst = []
        group_obj = df.groupby(col).mean()[response]

        for val in df[col]:
            new_lst.append(group_obj.loc[val])

        new_col_name = col + '_' + response
        df[new_col_name] = new_lst

    return df


def perform_feature_engineering(df, response):
    '''
    Generate the training and testing data for classification

    Input:
              df: pandas dataframe
              response: string of response name

    Output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    y = df['Churn']
    X = pd.DataFrame()
    df = encoder_helper(df, cat_columns, response)

    X[keep_cols] = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores
    report as image in images folder

    Input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    Output:
             None
    '''

    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(
        0.01, 0.6, str('Random Forest Test (below) Random Forest Train (above)'), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()

    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(
        0.01, 0.6, str('Logistic Regression Test (below) Logistic Regression Train (above)'), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    Creates and stores the feature importances in 'output_pth'

    Input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    Output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models

    Input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    Output:
              None
    '''

    # Instantiate sklearn modules
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # Set up grid search for models parameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Fit models
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # Predict on train data
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Predict on test data
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Store ROC curve with score
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_,
                   X_test,
                   y_test,
                   ax=ax,
                   alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()

    # Save the best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Save the model results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Save feature importances plot
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/feature_importances.png')


if __name__ == "__main__":
    # Import the data
    DATA = import_data("./data/bank_data.csv")

    # Perform EDA on the data
    perform_eda(DATA)

    # Split into training and testing data
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DATA, 'Churn')

    # Train models and store the results
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

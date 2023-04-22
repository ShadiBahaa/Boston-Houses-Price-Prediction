# imports
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from Constants import *

# Retrieving the boston house data as a dataframe from the csv file
boston_data = pd.read_csv(DATA_SET_FILE)

# A function to know info about the dataset to ensure if they don't contain any null or missing values


def discover_data():
    # Printing first 5 rows of dataset
    print(boston_data.head())
    # Printing the size of dataset
    print(boston_data.shape)
    # Checking the number of null values in data
    print(boston_data.isnull().sum())
    # Printing information about dataset
    print(boston_data.info())
    # Changing the name of the target column
    boston_data.rename(columns={CHANGE_FROM: CHANGE_TO}, inplace=True)
    # ensuring that the column name was changed
    print(boston_data.info())

# A function used to map the relation between dataset features as a correlation coefficients represented by a heatmap


def analyze_data():
    correlation_frame = boston_data.corr()
    # Plotting a 14 * 14 heatmap with values represented on each cell containing 2 decimal digits after point with
    # different blue degrees
    print(correlation_frame.shape)
    plt.figure(figsize=(WIDTH, HEIGHT))
    sbn.heatmap(correlation_frame, fmt=STRING_FORMATTING, annot=True, cmap=HEATMAP_COLOR)
    plt.show()

# Selecting the best features with the highest effect on the price, and the most important features in random forest.
# This helps to randomize features for no overfitting


def select_features():
    # features data for every row
    features_columns = boston_data.iloc[:, 0:13]
    # price for every row
    price_column = boston_data.iloc[:, 13]
    # Making it as integer for more accurate results
    price_column = np.round(price_column)
    # Selector for k best features
    suitable_features = SelectKBest(score_func=chi2, k=TOP_K_FEATURES)
    # fitting the price with the most suitable features
    result = suitable_features.fit(features_columns, price_column)
    # Score of each feature
    scores = pd.DataFrame(result.scores_)
    columns = pd.DataFrame(features_columns.columns)
    columns_scores = pd.concat([columns, scores], axis=COLUMN_AXIS)
    columns_scores.columns = ['Feature', 'Score']
    # Recognizing the most suitable features
    largest_dataframe = columns_scores.nlargest(TOP_K_FEATURES, "Score")
    print(largest_dataframe)
    # Using ExtraTreesClassifier for more control over over-fitting, getting random important features
    model = ExtraTreesClassifier()
    model.fit(features_columns, price_column)
    important_dataframe = pd.Series(model.feature_importances_, index=features_columns.columns).nlargest(TOP_K_FEATURES)
    print(important_dataframe)

# Training the data on RandomForestRegressor and test it


def train_and_validate():

    # I here chose random columns: some of the best features, some of the most important and some in between
    features_columns = boston_data.iloc[:, [5, -1, 9, 1, 11]]
    price_column = boston_data.iloc[:, 13]
    # Splitting the data to a training data and validation data
    features_train, features_test, price_train, price_test = train_test_split(features_columns, price_column,
                                                                              test_size=VALIDATION_SET_SIZE)
    # Fitting the data to a random forest regression model
    model = RandomForestRegressor()
    model.fit(features_train, price_train)
    price_predictions = model.predict(features_train)
    # Checking the accuracy
    print("Training accuracy: ", model.score(features_train, price_train)*100)
    print("Testing accuracy: ", model.score(features_test, price_test)*100)
    print("Model accuracy:", r2_score(price_column, model.predict(features_columns))*100)
    # Drawing a scatter plot as a relation between actual and predicted values
    plt.scatter(price_train, price_predictions)
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.show()


if __name__ == "__main__":
    discover_data()
    analyze_data()
    select_features()
    train_and_validate()

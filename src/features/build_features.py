import pandas as pd

# create dummy features
def create_dummy_vars(df):

    # Separate the input features and target variable
    X = df.drop('price', axis=1)
    y = df['price']

    return X, y
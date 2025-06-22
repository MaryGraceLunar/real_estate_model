# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Real Estate Solution Training',
#     author='Mary Grace Lunar',
#     license='',
# )

from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_LRmodel
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/final_realestate.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X, y = create_dummy_vars(df)

    # Train the Linear regression model
    model, X_test, y_test = train_LRmodel(X, y)

    # Evaluate the model
    mae = evaluate_model(model, X_test, y_test)
    print(f"Mean Square Error: {mae}")
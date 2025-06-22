# Import accuracy score
from sklearn.metrics import mean_absolute_error

# # Function to predict and evaluate
def evaluate_model(model, X_test, y_test):
    # Predict the loan eligibility on the testing set
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    mae = mean_absolute_error(y_pred, y_test)

    return mae
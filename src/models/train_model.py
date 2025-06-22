from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


# Function to train the model
def train_LRmodel(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Train the Linear regression model
    model = LinearRegression().fit(X_train, y_train)
    
    # Save the trained model
    with open('models/LRmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, X_test, y_test

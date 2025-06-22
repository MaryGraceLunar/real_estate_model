
import pandas as pd

def load_and_preprocess_data(data_path):
    
    # Import the data from 'credit.csv'
    df = pd.read_csv(data_path)

    # Impute all missing values in all the features
         #none to impute

    # Drop 'Loan_ID' variable from the data
        #none to drop

    return df

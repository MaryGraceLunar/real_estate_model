# real_estate_application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://real-estate-app-01.streamlit.app/)

password - streamlit

This application estimates the price of a property. The model aims to help users assess the property price by leveraging machine learning.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details such as credit history, loan amount, income, and other relevant factors.
- Real-time prediction of loan eligibility based on the trained model.
- Accessible via Streamlit Community Cloud.

## Dataset
The application is trained dataset that includes features like:
- Property Age
- Number of Bedromms
- Number of Bathrooms
- Basement
- Year Sold
- And other characteristic of a property.

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
The predictive model is trained using real estate dataset. The regression model used may include algorithm Random Forest, or Neural Network.

## Future Enhancements
* Extend the training and testing dataset to reflect the current market
* Adding visualizations to better represent user input and model predictions.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/_your-username_/credit_eligibility_application.git
   cd credit_eligibility_application

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

#### Thank you for using the Real Estate Estimator Application! Feel free to share your feedback.

CAFV Eligibility Predictor

Overview

The Clean Alternative Fuel Vehicle (CAFV) Eligibility Predictor is a machine learning application that determines whether an electric vehicle qualifies for CAFV eligibility based on various input parameters. The model uses logistic regression with hyperparameter tuning to improve accuracy.

Features

Loads and cleans electric vehicle data from a CSV file

Performs exploratory data analysis (EDA)

Encodes categorical features such as Electric Vehicle Type and Electric Utility

Trains a logistic regression model with GridSearchCV for hyperparameter tuning

Provides an interactive interface using ipywidgets for real-time predictions

Displays CAFV eligibility results along with confidence scores

Installation

To run this project, install the required dependencies:

pip install pandas numpy scikit-learn matplotlib ipywidgets

Dataset

The model is trained using an electric vehicle dataset containing the following key columns:

Electric Vehicle Type

Electric Range

Electric Utility

Clean Alternative Fuel Vehicle (CAFV) Eligibility

The target variable is is_CAFV, which is derived from the Clean Alternative Fuel Vehicle (CAFV) Eligibility column.

Model Training

The data is split into training and testing sets, and the features are scaled using StandardScaler. A logistic regression model is optimized using GridSearchCV with the following parameters:

C: [0.1, 1, 10, 100]

max_iter: [500, 1000, 2000]

Usage

Running the Model

Run the Python script to train the model and start the interactive predictor.

python script.py

Interactive Prediction

The model provides an interactive interface where users can select:

Electric Vehicle Type

Electric Range (miles)

Electric Utility

After clicking the "Predict" button, the system displays:

Eligibility status (CAFV Eligible or Not CAFV Eligible)

Confidence percentage

Input details

Example Output

✓ CAFV Eligible
Confidence: 61.53%

Input Details:
- Vehicle Type: Battery Electric Vehicle (BEV)
- Electric Range: 132 miles
- Electric Utility: BONNEVILLE POWER ADMINISTRATION

License

This project is open-source and available for modification and distribution.


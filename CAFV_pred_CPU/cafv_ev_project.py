
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

file_path = "Electric_Vehicle_Population_Data.csv"
df = pd.read_csv(file_path)

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    relevant_columns = [
        'Electric Vehicle Type',
        'Electric Range',
        'Electric Utility',
        'Clean Alternative Fuel Vehicle (CAFV) Eligibility'
    ]

    df = df[relevant_columns]

    df['Electric Range'].fillna(df['Electric Range'].median(), inplace=True)
    df['Electric Utility'].fillna("Unknown", inplace=True)
    df['Electric Vehicle Type'].fillna("Unknown", inplace=True)
    df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].fillna("Not Eligible", inplace=True)

    df['is_CAFV'] = (df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] ==
                     'Clean Alternative Fuel Vehicle Eligible').astype(int)

    # Encoding categorical variables
    label_encoders = {'Electric Vehicle Type': LabelEncoder(), 'Electric Utility': LabelEncoder()}
    for col in label_encoders:
        df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

    return df, label_encoders

def EDA(df):
    print("\nDataset Overview:")
    print("-----------------")
    print(f"Total number of records: {len(df)}")
    print("\nNull Values in Dataset:")
    print(df.isnull().sum())

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ev_cafv_counts = df.groupby(['Electric Vehicle Type', 'is_CAFV']).size().unstack()
    ev_cafv_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Vehicle Types vs CAFV Eligibility')
    ax1.set_xlabel('Vehicle Type')
    ax1.set_ylabel('Count')
    ax1.legend(['Not Eligible', 'Eligible'])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    df['Electric Range'].hist(bins=30, ax=ax2)
    ax2.set_title('Distribution of Electric Range')
    ax2.set_xlabel('Electric Range (miles)')
    ax2.set_ylabel('Count')

    df.boxplot(column='Electric Range', by='is_CAFV', ax=ax3)
    ax3.set_title('Electric Range by CAFV Eligibility')
    ax3.set_xlabel('CAFV Eligible')
    ax3.set_ylabel('Electric Range (miles)')

    utility_cafv = df.groupby('Electric Utility')['is_CAFV'].agg(['count', 'mean'])
    utility_cafv = utility_cafv.sort_values('count', ascending=False).head(10)
    utility_cafv['mean'].plot(kind='bar', ax=ax4)
    ax4.set_title('CAFV Eligibility Ratio by Top Utilities')
    ax4.set_xlabel('Electric Utility')
    ax4.set_ylabel('Eligible Ratio')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

df, label_encoders = load_and_clean_data(file_path)
EDA(df)

# Selecting features (including encoded categorical columns)
feature_columns = ['Electric Range', 'Electric Vehicle Type_encoded', 'Electric Utility_encoded']
X = df[feature_columns]
y = df['is_CAFV']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training with proper features
param_grid = {'C': [0.1, 1, 10, 100], 'max_iter': [500, 1000, 2000]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Optimized Logistic Regression Model Accuracy:", accuracy)

def predict_cafv(df, model, scaler, label_encoders):
    ev_type = widgets.Dropdown(
        description='EV Type:',
        options=sorted(df['Electric Vehicle Type'].dropna().unique()),
        style={'description_width': 'initial'}
    )

    range_input = widgets.FloatSlider(
        description='Electric Range (miles):',
        min=0,
        max=500,
        step=1,
        value=200,
        style={'description_width': 'initial'}
    )

    utility = widgets.Dropdown(
        description='Electric Utility:',
        options=sorted(df['Electric Utility'].dropna().unique()),
        style={'description_width': 'initial'}
    )

    predict_button = widgets.Button(description='Predict', button_style='info')
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            clear_output()
            input_data = pd.DataFrame({
                'Electric Range': [range_input.value],
                'Electric Vehicle Type_encoded': [label_encoders['Electric Vehicle Type'].transform([ev_type.value])[0]],
                'Electric Utility_encoded': [label_encoders['Electric Utility'].transform([utility.value])[0]]
            })

            X_scaled = scaler.transform(input_data)
            prediction = model.predict(X_scaled)[0]
            result_text = '✓ CAFV Eligible' if prediction > 0.5 else '✗ Not CAFV Eligible'

            display(HTML(f"""
                <div style='padding: 10px; border-radius: 5px; background-color: {'#28a745' if prediction > 0.5 else '#dc3545'};'>
                    <h3>{result_text}</h3>
                </div>
            """))

    predict_button.on_click(on_button_clicked)
    display(ev_type, range_input, utility, predict_button, output)

# Call prediction function with updated pre-processing
predict_cafv(df, best_model, scaler, label_encoders)

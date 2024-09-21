import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv(r"student dropout.csv")

# Convert 'Dropped_Out' to binary (assuming 'False' and 'True' as strings)
data['Dropped_Out'] = data['Dropped_Out'].apply(lambda x: 1 if x == 'True' else 0)

# Check the current class distribution
print("Original class distribution:")
print(data['Dropped_Out'].value_counts())

# Simulate dropout cases to balance the dataset
num_dropouts = int(0.2 * len(data))
dropout_indices = np.random.choice(data.index, num_dropouts, replace=False)
data.loc[dropout_indices, 'Dropped_Out'] = 1

# Verify the new class distribution
print("New class distribution after simulating dropouts:")
print(data['Dropped_Out'].value_counts())

# Proceed with data preprocessing
X = data.drop(columns=['Dropped_Out'])  # Features
y = data['Dropped_Out']  # Target

# Label encoding for binary categorical features
binary_columns = ['Gender', 'Address', 'Family_Size', 'Parental_Status', 
                  'School_Support', 'Family_Support', 'Extra_Paid_Class', 
                  'Extra_Curricular_Activities', 'Attended_Nursery', 
                  'Wants_Higher_Education', 'Internet_Access', 'In_Relationship']

le = LabelEncoder()
for col in binary_columns:
    X[col] = le.fit_transform(X[col])

# One-hot encoding for categorical features with multiple categories
X = pd.get_dummies(X, columns=['School', 'Mother_Job', 'Father_Job', 
                               'Reason_for_Choosing_School', 'Guardian'], drop_first=True)

# Save the list of features for later use
feature_names = X.columns.tolist()

# Split data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(conf_matrix)

# Streamlit app
# Load the model and features

st.title("Student Dropout Prediction App")

# User inputs for predicting student dropout
school = st.selectbox('School', ['GP', 'MS'])
gender = st.selectbox('Gender', ['M', 'F'])
age = st.slider('Age', 15, 22, 18)
address = st.selectbox('Home Address', ['U', 'R'])
famsize = st.selectbox('Family Size', ['GT3', 'LE3'])
pstatus = st.selectbox('Parental Status', ['T', 'A'])
medu = st.slider('Mother\'s Education (0=none, 4=higher education)', 0, 4, 2)
fedu = st.slider('Father\'s Education (0=none, 4=higher education)', 0, 4, 2)
mjob = st.selectbox("Mother's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
fjob = st.selectbox("Father's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
reason = st.selectbox('Reason for Choosing School', ['home', 'reputation', 'course', 'other'])
guardian = st.selectbox('Guardian', ['mother', 'father', 'other'])
traveltime = st.slider('Travel Time to School (1= <15 min, 4= >60 min)', 1, 4, 1)
studytime = st.slider('Study Time (1= <2 hrs, 4= >10 hrs)', 1, 4, 2)
failures = st.number_input('Number of Past Class Failures', min_value=0, max_value=4, value=0)
schoolsup = st.selectbox('School Support', ['yes', 'no'])
famsup = st.selectbox('Family Support', ['yes', 'no'])
paid = st.selectbox('Extra Paid Classes', ['yes', 'no'])
activities = st.selectbox('Extra-curricular Activities', ['yes', 'no'])
nursery = st.selectbox('Attended Nursery', ['yes', 'no'])
higher = st.selectbox('Wants Higher Education', ['yes', 'no'])
internet = st.selectbox('Internet Access at Home', ['yes', 'no'])
romantic = st.selectbox('In a Romantic Relationship', ['yes', 'no'])
famrel = st.slider('Family Relationship Quality (1=very bad, 5=excellent)', 1, 5, 4)
freetime = st.slider('Free Time after School (1=very low, 5=very high)', 1, 5, 3)
goout = st.slider('Going Out with Friends (1=very low, 5=very high)', 1, 5, 3)
dalc = st.slider('Workday Alcohol Consumption (1=very low, 5=very high)', 1, 5, 1)
walc = st.slider('Weekend Alcohol Consumption (1=very low, 5=very high)', 1, 5, 2)
health = st.slider('Current Health Status (1=very bad, 5=very good)', 1, 5, 5)
absences = st.number_input('Number of School Absences', min_value=0, max_value=93, value=0)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Mother_Education': [medu],
    'Father_Education': [fedu],
    'Travel_Time': [traveltime],
    'Study_Time': [studytime],
    'Failures': [failures],
    'Family_Relationship_Quality': [famrel],
    'Free_Time': [freetime],
    'Going_Out': [goout],
    'Workday_Alcohol_Consumption': [dalc],
    'Weekend_Alcohol_Consumption': [walc],
    'Health_Status': [health],
    'Number_of_Absences': [absences],
    'Gender': [gender],
    'Address': [address],
    'Family_Size': [famsize],
    'Parental_Status': [pstatus],
    'School_Support': [schoolsup],
    'Family_Support': [famsup],
    'Extra_Paid_Class': [paid],
    'Extra_Curricular_Activities': [activities],
    'Attended_Nursery': [nursery],
    'Wants_Higher_Education': [higher],
    'Internet_Access': [internet],
    'In_Relationship': [romantic],
    'School': [school],
    'Mother_Job': [mjob],
    'Father_Job': [fjob],
    'Reason_for_Choosing_School': [reason],
    'Guardian': [guardian],
})

# Preprocess the input data
for col in binary_columns:
    input_data[col] = le.fit_transform(input_data[col])

input_data = pd.get_dummies(input_data, columns=['School', 'Mother_Job', 'Father_Job', 
                                                 'Reason_for_Choosing_School', 'Guardian'], drop_first=True)

# Align input data with model's features
features = load_features()
input_data = input_data.reindex(columns=features, fill_value=0)

# Load the model
model = load_model()

# Predict and display results
if st.button('Predict'):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of dropout
    
    # Display prediction based on the result
    if prediction[0] == 1:
        st.warning(f"The student is at risk of dropping out with a probability of {probability:.2f}.")
    else:
        st.success(f"The student is not at risk of dropping out with a probability of {1 - probability:.2f}.")

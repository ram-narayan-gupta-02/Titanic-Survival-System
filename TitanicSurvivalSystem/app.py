# For the run file on browser
# python -m streamlit run app.py 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Titanic dataset two different ways
# df = pd.read_csv("Train.csv")  
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Preparing dataset columns for analysis
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)  

# Convert categorical data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Display the input variables (features) and the target output (labels)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

# Separate the data for model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# 🎨 Streamlit UI for your model
st.title("🚢 Titanic Survival Prediction System")

# User input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.number_input("Fare Paid", 0.0, 500.0, 7.25)
embarked_q = st.checkbox("Embarked at Queenstown (Q)")
embarked_s = st.checkbox("Embarked at Southampton (S)")
embarked_c = st.checkbox("Embarked at Cherbourg (C)")

# Convert inputs
sex = 0 if sex == "Male" else 1
embarked_q = 1 if embarked_q else 0
embarked_s = 1 if embarked_s else 0
embarked_c = 1 if embarked_c else 0

# Prediction button
if st.button("Predict"):
    # Prepare input data
    user_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_q, embarked_s]])
    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)
    result = "Survived ✅" if prediction[0] == 1 else "Not Survived ❌"
    
    # Display result
    st.subheader("Prediction Result:")
    st.success(result)

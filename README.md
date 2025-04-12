# Titanic Survival Prediction System

## 📌 Project Overview
This project is a **machine learning-based system** that predicts whether a passenger on the Titanic would survive or not based on various features such as age, gender, ticket class, fare, and embarkation point.

The model is built using **Python, pandas, scikit-learn, and Streamlit** for deployment, providing an interactive web interface for predictions.
![Uploading ChatGPT Image Mar 30, 2025, 02_16_41 PM.png…]()

## 🚀 Features
- **Data Preprocessing**: Handles missing values and categorical variables.
- **Exploratory Data Analysis (EDA)**: Visualizes key patterns and insights.
- **Feature Engineering**: Transforms data for better model performance.
- **Machine Learning Model**: Implements Logistic Regression for prediction.
- **Web App with Streamlit**: Allows users to input passenger details and get survival predictions.

---

## 📂 Project Structure
```
TitanicSurvivalSystem/
│── data/
│   ├── train.csv
│── app.py
│── README.md
```
- **data/**: Contains Titanic dataset.
- **app.py**: Streamlit app for prediction.
- **run_streamlit.py**: Script to launch the web app.
- **README.md**: Project documentation.
---

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/TitanicSurvivalSystem.git
cd TitanicSurvivalSystem
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv  # Windows
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Data Preprocessing & EDA
The dataset contains:
- **PassengerId** (Unique ID)
- **Survived** (Target: 1 = Survived, 0 = Not Survived)
- **Pclass** (Ticket class)
- **Name, Sex, Age**
- **SibSp, Parch** (Family relations)
- **Ticket, Fare, Cabin, Embarked**

### 🔹 Handling Missing Data
- **Age**: Filled with median age.
- **Embarked**: Filled with mode (most common value).
- **Cabin**: Dropped due to excessive missing values.

### 🔹 Encoding Categorical Variables
- **Sex**: Converted to numeric (0 = Male, 1 = Female).
- **Embarked**: One-hot encoding for categorical values (C, Q, S).

---

## 🔍 Model Training
### **Algorithm Used**: Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```
### **Model Performance**
- Accuracy: **81%**
- Precision, Recall, F1-Score: Evaluated for class imbalance.

---

## 🖥️ Running the Web App
### **Run Streamlit App**
```bash
streamlit run app.py
```
### **Alternative: Run via Python Script**
```bash
python -m streamlit run app.py
```
### **Access in Browser**
Once running, open in your browser.

---

## 🛠️ Future Improvements
- Implement more models (Random Forest, SVM, XGBoost).
- Add a graphical dashboard for EDA.
- Deploy the app using **Heroku or AWS**.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing
Feel free to contribute by submitting a pull request or reporting issues!

---

## 📩 Contact
📧 Email: [ramnrngupta@gmail.com](mailto:ramnrngupta@gmail.com)
📌 GitHub: [ram-narayan-gupta-02](https://github.com/ram-narayan-gupta-02)


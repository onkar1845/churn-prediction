# 📊 Customer Churn Prediction System

## 🚀 Overview

This project predicts whether a customer is likely to churn using machine learning.
It helps businesses identify high-risk customers and take proactive retention actions.

---

## 🧠 Key Features

* End-to-end ML pipeline (data → model → deployment)
* Data preprocessing & feature engineering
* Exploratory Data Analysis (EDA) with insights
* Multiple models: Logistic Regression, Random Forest, XGBoost
* Class imbalance handling (improved churn recall)
* Streamlit web app for real-time prediction
* Probability-based output (confidence score)

---

## 📊 Dataset

* Telco Customer Churn Dataset (~7000 records)
* Features include:

  * Customer demographics
  * Services subscribed
  * Billing & payment details
  * Tenure (customer duration)

---

## 📈 Model Performance

| Model                        | Accuracy | Recall (Churn) |
| ---------------------------- | -------- | -------------- |
| Logistic Regression          | 0.78     | 0.52           |
| Balanced Logistic Regression | 0.73     | **0.79 🔥**    |
| Random Forest                | 0.78     | 0.46           |
| XGBoost                      | 0.74     | 0.65           |

👉 Final Model: **Balanced Logistic Regression (best for churn detection)**

---

## 🧠 Key Insights

* Customers with low tenure are more likely to churn
* High monthly charges increase churn probability
* Month-to-month contracts have highest churn risk
* Long-term contracts reduce churn significantly

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn
* Streamlit

---

## 🖥️ Demo (Streamlit App)

### ▶️ Run Locally

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 📸 App Screenshot

(Add your Streamlit UI screenshot here)

---

## 📂 Project Structure

```
churn-prediction/
│
├── src/
│   └── data_loading.py
├── app/
│   └── app.py
├── models/
├── data/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📌 Future Improvements

* Add more feature inputs in UI
* Deploy on cloud (Render / AWS)
* Hyperparameter tuning
* Model explainability (SHAP)

---

## 👨‍💻 Author

Final Year Computer Science Student
Machine Learning Enthusiast 🚀

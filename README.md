# 🏦 Loan Default Risk Prediction using Decision Tree Classifier

This project predicts the risk of loan default using a **🌳 Decision Tree Classifier**, trained on the **Home Credit Default Risk** dataset (Kaggle).  

📊 An interactive **Streamlit** app is included for testing the model with custom inputs.

---

## 📂 **Project Structure**

Explore how everything fits together below 👇

<details>
<summary>📁 <strong>FINAL PROJECT/</strong> (click to expand)</summary>

├── 📄 application_train.csv
│ └─ Raw input dataset used for model training
│
├── 📄 engineered_data.csv
│ └─ Preprocessed & cleaned data for better learning
│
├── 🧠 best_decision_tree_model.pkl
│ └─ Saved Decision Tree model (after tuning)
│
├── 🚀 app.py
│ └─ Interactive Streamlit app to test predictions
│
├── 📓 project.ipynb
│ └─ Full EDA + training + pruning + evaluation steps
│
├── 📦 requirements.txt
│ └─ List of dependencies to install the project
│
└── 📘 README.md
└─ You're reading it right now 😄

bash
Copy
Edit

</details>

## 📊 **Dataset**

- **Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- **File:** `application_train.csv`
- **Rows:** ~307,000
- **Features:** 120+ raw → ~180+ after feature engineering
- **Target:** `TARGET` (1 = Defaulted, 0 = Paid Back)

---

## 📜 **Workflow**

**Data Understanding & Preprocessing**
- Removed columns with >50% missing values.
- Filled numeric missing with median, categorical with mode.
- Encoded categorical features (Label Encoding, One-Hot Encoding).
- Final shape: ~307,511 rows × ~181 columns.

**Feature Engineering**
- Age and employment: `AGE`, `YEARS_EMPLOYED`, `AGE_EMPLOYED_RATIO`
- Financial ratios: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `ANNUITY_CREDIT_RATIO`
- Social circle, contact, flags, and regional features.
- Outlier detection & skewness check.

**Model Training**
- Split: Train / Validation / Test
- Tuned Decision Tree (criterion, max_depth).
- Unpruned tree: ~73% accuracy, slight overfitting.
- Pruned tree & Random Forest used for comparison.

**Deployment**
- Streamlit app for live predictions.
- User enters key features.
- Model uses full feature pipeline for prediction.
- Output: Probability & class (Default or No Default).


**For DataSet You Can Mail Me At abdullah.attique.2005@gmail.com**
---

## 🚀 **How to Run**

1️⃣ Install dependencies:  
```bash
pip install -r requirements.txt

2️⃣ Run the Streamlit app:

streamlit run app.py

```
---

## 👥 **Team Members**
This project is brought to life by the talented team of:

- Abdullah Attique 🚀
- Minahil Rizwan ✨
- Muhammad Nade Ali 🖥️

---

## 📢 **License**

© 2025 [Abdullah Attique]. All rights reserved.

This project is developed for educational purposes only and follows the Honor Policy of the institution to promote academic integrity.  
Unauthorized commercial use, reproduction, or distribution is strictly prohibited.

---


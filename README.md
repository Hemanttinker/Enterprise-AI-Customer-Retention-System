# 🚀 Enterprise AI: Predictive Customer Retention & ML Pipeline

An end-to-end Machine Learning and Business Intelligence system that processes over 1 Million+ retail transaction logs, segments customers using RFM analysis, and utilizes an unsupervised and supervised ML pipeline to predict real-time customer churn risk.

📊 **Interactive Dashboard:** Built with Streamlit to provide production-ready analytics for retention marketing strategies.

---

## 🧠 System Architecture & Core Modules

### 1. Data Pipeline & Financial Engineering
* **Scale:** Processed `1,067,371` raw transaction records from a large UK-based online retail dataset.
* **Data Hygiene:** Handled structural data anomalies, eliminated missing categorical identities (`Customer ID`), managed stock returns/cancellations (negative quantities/prices), and engineered a unified financial metric (`TotalAmount`).

### 2. Analytical Segmentation (RFM Framework)
* Engineered cohort metrics across three strategic dimensions:
    * **Recency:** Days elapsed since the last transaction.
    * **Frequency:** Total volume of distinct invoice logs.
    * **Monetary:** Cumulative financial yield per unique consumer.
* Mapped behavioral attributes into 10 industry-standard customer archetypes (e.g., *Champions*, *Loyal Customers*, *At Risk*, *Hibernating*).

### 3. Predictive Machine Learning (Supervised Framework)
* **Feature Scaling:** Standardized dense numerical ranges using a robust `StandardScaler` to remove skewness.
* **Model Architecture:** Trained a high-capacity **Random Forest Classifier** with balanced class weights to account for dynamic churn boundaries.
* **Target Labeling:** Programmed a 90-day activity horizon threshold to automatically detect inactive accounts.

### 4. Interactive Enterprise Dashboard
* Implements a real-time predictive service where selecting any active customer ID instantly displays their cohort segment alongside an AI-calculated churn risk probability percentage.

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, Preprocessing)
* **Data Visualization:** Matplotlib, Seaborn
* **Application UI:** Streamlit Web Framework
* **Model Serialization:** Pickle Protocol

---

## 🏃‍♂️ How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Enterprise-AI-Customer-Retention-System.git](https://github.com/YOUR_USERNAME/Enterprise-AI-Customer-Retention-System.git)
   pip install -r requirements.text
   streamlit run app.py

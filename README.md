# 🫀 Heart-AI: Intelligent Cardiovascular Diagnostic System

[![Live Demo](https://img.shields.io/badge/Demo-Live_Website-00f2fe?style=for-the-badge&logo=vercel)](https://heart-disease-pred-4z15.onrender.com)
[![Python Flask](https://img.shields.io/badge/Backend-Flask-black?style=for-the-badge&logo=flask)](#)
[![TensorFlow](https://img.shields.io/badge/AI_Engine-TensorFlow-orange?style=for-the-badge&logo=tensorflow)](#)

Heart-AI is an advanced, dual-engine artificial intelligence platform designed to assist in the early detection and diagnosis of cardiovascular anomalies. Wrapped in a modern, ultra-aesthetic dark glassmorphism UI, it bridges the gap between complex machine learning models and clinical usability.

---

## ✨ Key Features

* 📊 **Clinical Vitals Engine (Tabular Data):
* ** Utilizes an **XGBoost** machine learning classifier to analyze 13 critical clinical parameters (Age, BP, Cholesterol, ECG results, etc.) to predict the immediate risk of heart disease.
* 📸 **ECG Vision AI (Image Data):
* ** Employs a Custom **Convolutional Neural Network (CNN)** built with TensorFlow/Keras to visually analyze ECG scan images and classify anomalies like Myocardial Infarction or Abnormal Heartbeats.
* 📈 **Interactive Clinical Dashboard:
* ** Features dynamic, real-time visual comparisons of patient vitals against healthy medical thresholds using **Chart.js**.
* 📄 **Automated PDF Medical Reports:
* ** Generates professional, hospital-grade diagnostic PDFs via **FPDF**, complete with AI confidence scores, findings, and tailored clinical recommendations.
* 🗄️ **Secure Patient History Archive:
* ** A dedicated **SQLite** database tracks all historical assessments, allowing medical professionals to revisit and download past reports instantly.

---

## 🛠️ Technology Stack

### Frontend (UI/UX)
* HTML5 / CSS3 (Dark Glassmorphism Theme)
* JavaScript (DOM Manipulation & Validation)
* **Chart.js** (Data Visualization)
* Plus Jakarta Sans & Space Grotesk (Typography)

### Backend & Database
* **Python 3.x**
* **Flask** (Web Framework)
* **Flask-SQLAlchemy** (SQLite Database Management)
* **FPDF** (Automated PDF Generation)

### Machine Learning & AI
* **TensorFlow / Keras** (Deep Learning - CNN)
* **XGBoost / Scikit-Learn** (Predictive Modeling)
* **Pandas & NumPy** (Data Processing)
* **Joblib** (Model Serialization)

---

## 🚀 Live Demonstration

Experience the platform live: **[Click here to visit Heart-AI](https://heart-disease-pred-4z15.onrender.com)**

### 🧪 Test Cases for Evaluators:
To test the system's accuracy, you can use the following parameters in the **Clinical Vitals** module:
* **High Risk (Danger):** Age: 72 | CP: 3 | BP: 185 | Chol: 340 | FBS: 1 | Max HR: 88 | ExAng: 1
* **Low Risk (Healthy):** Age: 24 | CP: 0 | BP: 118 | Chol: 185 | FBS: 0 | Max HR: 182 | ExAng: 0

---

## 💻 Local Installation & Setup

Want to run this project locally? Follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/amansingh-gh/heart-ai.git]
   cd heart-ai


2. **Create a virtual environment (Recommended):**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install the required dependencies:**
pip install -r requirements.txt

4. **Ensure the ML models are in the root directory:**
vitals_model.pkl
ecg_vision_model.h5

5. **Run the Flask application:**
python app.py

The application will be live at http://127.0.0.1:5000

⚠️ Disclaimer
This application is developed for educational and portfolio purposes. The AI-generated reports are not a substitute for professional medical advice, diagnosis, or treatment.

Crafted with ❤️ by Aman Singh

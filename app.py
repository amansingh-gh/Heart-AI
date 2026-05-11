import os
import numpy as np
import joblib 
import pandas as pd
import random
from datetime import datetime
from flask import Flask, render_template, request, redirect, send_file, make_response
from flask_sqlalchemy import SQLAlchemy

# ML Tools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# PDF Tool
from fpdf import FPDF

app = Flask(__name__)

# DATABASE CONFIGURATION
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class PatientHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    diagnostic_type = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# MODEL LOADING
ecg_model = load_model('ecg_vision_model.h5')
vitals_model = joblib.load('vitals_model.pkl')

ecg_classes = [
    'Unrecognized Scan',         # 0
    'Abnormal Heartbeat',        # 1
    'Myocardial Infarction',     # 2
    'Normal',                    # 3
    'Post MI History'            # 4
]

# ROUTES (PAGES)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('landingPage.html')

@app.route('/history')
def history():
    records = PatientHistory.query.order_by(PatientHistory.date.desc()).all()
    return render_template('history.html', records=records)

@app.route('/predict_image_page')
def predict_image_page():
    return render_template('predict_image.html')

@app.route('/predict_vitals', methods=['GET', 'POST'])
def vital_page():
    if request.method == 'POST':
        try:
            name = request.form.get('patient_name', 'Unknown')
            age = float(request.form.get('age') or 0)
            sex = float(request.form.get('sex') or 0)
            cp = float(request.form.get('cp') or 0)
            trestbps = float(request.form.get('trestbps') or 0)
            chol = float(request.form.get('chol') or 0)
            fbs = float(request.form.get('fbs') or 0)
            restecg = float(request.form.get('restecg') or 0)
            thalach = float(request.form.get('thalach') or 0)
            exang = float(request.form.get('exang') or 0)
            oldpeak = float(request.form.get('oldpeak') or 0)
            slope = float(request.form.get('slope') or 0)
            ca = float(request.form.get('ca') or 0)
            thal = float(request.form.get('thal') or 0)

            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                      thalach, exang, oldpeak, slope, ca, thal]], 
                                    columns=feature_names)

            model_prediction = vitals_model.predict(input_df)[0]
            probability_array = vitals_model.predict_proba(input_df)[0]

            conf = round(max(probability_array) * 100, 2)
            
            # FIXED LOGIC: 1 = High Risk, 0 = Low Risk
            if model_prediction == 0:
                res = "High Risk of Heart Disease"
            else:
                res = "Low Risk of Heart Disease"

            new_rec = PatientHistory(
                name=name, 
                age=int(age), 
                prediction_result=res, 
                probability=conf, 
                diagnostic_type="Clinical Vitals AI"
            )
            db.session.add(new_rec)
            db.session.commit()

            return render_template('vitals_result.html', 
                                   prediction_text=res, 
                                   probability=conf,
                                   name=name, age=age, 
                                   bp=trestbps, chol=chol, hr=thalach)
            
        except Exception as e:
            print(f"Error: {e}")
            return render_template('predict_vitals.html', prediction_text="Error processing data.", probability=0)

    return render_template('predict_vitals.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'ecg_image' not in request.files:
        return redirect('/predict_image_page')
    
    file = request.files['ecg_image']
    name = request.form.get('patient_name', 'Unknown')
    age = request.form.get('age', 0)

    if file and file.filename != '':
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
            
        path = os.path.join('static/uploads', file.filename)
        file.save(path)
        
        img = image.load_img(path, target_size=(224, 224))
        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        
        preds = ecg_model.predict(img_arr)
        idx = np.argmax(preds)
        conf = round(np.max(preds) * 100, 2)
        res = ecg_classes[idx]
        
        new_rec = PatientHistory(
            name=name, age=int(age), 
            prediction_result=res, probability=conf, 
            diagnostic_type="ECG Vision AI"
        )
        db.session.add(new_rec)
        db.session.commit()
        
        return render_template('predict_image.html', 
                               prediction_text=res, probability=conf, 
                               user_image=path, p_name=name, p_age=age)
                               
    return redirect('/predict_image_page')

@app.route('/download_report/<name>/<age>/<prediction_text>/<probability>')
def download_report(name, age, prediction_text, probability):
    pdf = FPDF()
    pdf.add_page()
    
    PRIMARY = (0, 180, 216)
    DARK = (44, 62, 80)
    DANGER = (231, 76, 60)
    SUCCESS = (39, 174, 96)
    
    # Header
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 10, "HEART-AI DIAGNOSTIC CENTER", ln=True, align='C')
    pdf.ln(10)
    
    # Patient Info
    report_id = f"HAI-{random.randint(10000, 99999)}"
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(*DARK)
    pdf.cell(100, 8, f"PATIENT NAME: {name.upper()}")
    pdf.cell(90, 8, f"REPORT ID: {report_id}", align='R', ln=True)
    pdf.cell(100, 8, f"AGE: {age} Years")
    pdf.cell(90, 8, f"DATE: {datetime.now().strftime('%d %B %Y')}", align='R', ln=True)
    pdf.line(10, 50, 200, 50)
    pdf.ln(15)
    
    # Verdict
    is_high_risk = "High" in prediction_text or "Abnormal" in prediction_text or "Infarction" in prediction_text
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. AI DIAGNOSTIC VERDICT", ln=True)
    
    if is_high_risk:
        pdf.set_text_color(*DANGER)
    else:
        pdf.set_text_color(*SUCCESS)
    pdf.cell(0, 8, f"STATUS: {prediction_text.upper()}", ln=True)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, f"AI Confidence Score: {probability}%", ln=True)
    pdf.ln(10)

    # Findings & Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. CLINICAL RECOMMENDATIONS", ln=True)
    pdf.set_font("Arial", '', 11)
    if is_high_risk:
        rec = "URGENT: Consult a Cardiologist immediately. AI indicates critical cardiovascular distress markers."
    else:
        rec = "Routine health maintained. Follow a balanced diet and regular exercise."
    pdf.multi_cell(0, 7, rec)
    
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers.set('Content-Disposition', f'attachment; filename={name}_Report.pdf')
    response.headers.set('Content-Type', 'application/pdf')
    return response

if __name__ == '__main__':
    app.run(debug=True)
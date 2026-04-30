import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__)

# ==========================================
# DATABASE CONFIGURATION
# ==========================================
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

# ==========================================
# MODEL LOADING
# ==========================================
# Make sure your model file is in the same folder, or update the path (e.g., 'Models/ecg_vision_model_light.h5')
ecg_model = load_model('ecg_vision_model.h5')

# The exact order based on alphanumeric sorting of folders
ecg_classes = [
    'Unrecognized_Scan',         # 0
    'Abnormal Heartbeat',        # 1
    'Myocardial Infarction',     # 2
    'Normal',                    # 3
    'Post MI History'            # 4
]

# ==========================================
# ROUTES (PAGES)
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    records = PatientHistory.query.order_by(PatientHistory.date.desc()).all()
    return render_template('history.html', records=records)

@app.route('/predict_image_page')
def predict_image_page():
    return render_template('predict_image.html')


# ==========================================
# VITALS PREDICTION LOGIC (FIXED ROUTING)
# ==========================================
@app.route('/predict_vitals', methods=['GET', 'POST'])
def vital_page():
    if request.method == 'POST':
        # 1. Fetching all data from the HTML form
        name = request.form.get('patient_name', 'Unknown')
        age = float(request.form.get('age', 0))
        sex = float(request.form.get('sex', 0))
        cp = float(request.form.get('cp', 0))
        trestbps = float(request.form.get('trestbps', 0))
        chol = float(request.form.get('chol', 0))
        fbs = float(request.form.get('fbs', 0))
        restecg = float(request.form.get('restecg', 0))
        thalach = float(request.form.get('thalach', 0))
        exang = float(request.form.get('exang', 0))
        oldpeak = float(request.form.get('oldpeak', 0))
        slope = float(request.form.get('slope', 0))
        ca = float(request.form.get('ca', 0))
        thal = float(request.form.get('thal', 0))

        # ----------------------------------------------------
        # 2. YOUR ML MODEL LOGIC WILL GO HERE IN FUTURE
        # Example: model = joblib.load('vitals_model.pkl')
        # prediction = model.predict([[age, sex, cp, trestbps...]])
        # ----------------------------------------------------

        # DUMMY PREDICTION (To check if routing and UI are working perfectly)
        res = "High Risk of Heart Disease" 
        conf = 88.5 

        # Save Vitals prediction to Database
        new_rec = PatientHistory(
            name=name, 
            age=int(age), 
            prediction_result=res, 
            probability=conf, 
            diagnostic_type="Clinical Vitals AI"
        )
        db.session.add(new_rec)
        db.session.commit()

        # Send data back to the same page to display results
        return render_template('predict_vitals.html', prediction_text=res, probability=conf)

    # If it's a GET request (just opening the page)
    return render_template('predict_vitals.html')


# ==========================================
# ECG AI PREDICTION LOGIC
# ==========================================
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
        
        # Standard Custom CNN Preprocessing
        img = image.load_img(path, target_size=(224, 224))
        img_arr = image.img_to_array(img) / 255.0  # Proper scaling for Custom CNN
        img_arr = np.expand_dims(img_arr, axis=0)
        
        preds = ecg_model.predict(img_arr)
        idx = np.argmax(preds)
        conf = round(np.max(preds) * 100, 2)
        
        res = ecg_classes[idx]
        
        if res == 'Unrecognized_Scan':
            res = 'Unrecognized Scan'
            
        new_rec = PatientHistory(
            name=name, 
            age=int(age), 
            prediction_result=res, 
            probability=conf, 
            diagnostic_type="ECG Vision AI"
        )
        db.session.add(new_rec)
        db.session.commit()
        
        return render_template('predict_image.html', 
                               prediction_text=res, probability=conf, 
                               user_image=path, p_name=name, p_age=age)
                               
    return redirect('/predict_image_page')

# ==========================================
# PDF DOWNLOAD LOGIC
# ==========================================
@app.route('/download_report/<name>/<age>/<result>/<confidence>')
def download_report(name, age, result, confidence):
    if not os.path.exists('static/reports'):
        os.makedirs('static/reports')
        
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(200, 10, txt="HEART-AI CLINICAL DIAGNOSTIC REPORT", ln=True, align='C')
    pdf.line(10, 20, 200, 20)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Age: {age} Years", ln=True)
    pdf.cell(200, 10, txt=f"Date of Scan: {datetime.utcnow().strftime('%Y-%m-%d')}", ln=True)
    pdf.line(10, 55, 200, 55)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="AI Vision Findings:", ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    if 'Normal' in result or 'Low' in result:
        pdf.set_text_color(39, 174, 96)
    elif 'Unrecognized' in result:
        pdf.set_text_color(230, 126, 34)
    else:
        pdf.set_text_color(231, 76, 60)
        
    pdf.cell(200, 10, txt=f"Diagnosis: {result}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"AI Confidence Score: {confidence}%", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Clinical Recommendation:", ln=True)
    pdf.set_font("Arial", size=11)
    
    if 'Normal' in result or 'Low' in result:
        pdf.multi_cell(0, 8, txt="No immediate cardiac concerns detected. Routine checkups are advised to maintain cardiovascular health.")
    elif 'Unrecognized' in result:
        pdf.multi_cell(0, 8, txt="The uploaded image could not be confidently identified as an ECG scan. Please upload a valid medical document for analysis.")
    else:
        pdf.multi_cell(0, 8, txt="Abnormal patterns detected. Please consult a Cardiologist immediately for clinical correlation, further testing, and medical management.")
        
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5, txt="Disclaimer: This report is generated by an Artificial Intelligence system (Heart-AI) intended for research purposes. It does not replace professional medical advice, diagnosis, or treatment.")
    
    report_path = f"static/reports/{name}_Diagnostic_Report.pdf"
    pdf.output(report_path)
    
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
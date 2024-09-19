from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import pandas as pd
import numpy as np

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo desde la ruta especificada
MODEL_PATH = 'models/best_model.pkl'
best_model = joblib.load(MODEL_PATH)

# Ruta para renderizar el formulario
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Ruta para manejar la solicitud de predicción
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Recoge todos los datos del formulario aquí y conviértelos al tipo adecuado.
        input_dict = {
            'maritalStatus': int(request.form['maritalStatus']),
            'applicationMode': int(request.form['applicationMode']),
            'applicationOrder': int(request.form['applicationOrder']),
            'course': int(request.form['course']),
            'attendance': int(request.form['attendance']),
            'previousQualification': int(request.form['previousQualification']),
            'nationality': int(request.form['nationality']),
            'motherQualification': int(request.form['motherQualification']),
            'fatherQualification': int(request.form['fatherQualification']),
            'motherOccupation': int(request.form['motherOccupation']),
            'fatherOccupation': int(request.form['fatherOccupation']),
            'displaced': int(request.form['displaced']),
            'educationalNeeds': int(request.form['educationalNeeds']),
            'debtor': int(request.form['debtor']),
            'tuitionUpToDate': int(request.form['tuitionUpToDate']),
            'gender': int(request.form['gender']),
            'scholarshipHolder': int(request.form['scholarshipHolder']),
            'ageAtEnrollment': int(request.form['ageAtEnrollment']),
            'international': int(request.form['international']),
            'units1stCredited': int(request.form['units1stCredited']),
            'units1stEnrolled': int(request.form['units1stEnrolled']),
            'units1stEvaluations': int(request.form['units1stEvaluations']),
            'units1stApproved': int(request.form['units1stApproved']),
            'units1stGrade': float(request.form['units1stGrade']),
            'units1stWithoutEvaluations': int(request.form['units1stWithoutEvaluations']),
            'units2ndCredited': int(request.form['units2ndCredited']),
            'units2ndEnrolled': int(request.form['units2ndEnrolled']),
            'units2ndEvaluations': int(request.form['units2ndEvaluations']),
            'units2ndApproved': int(request.form['units2ndApproved']),
            'units2ndGrade': float(request.form['units2ndGrade']),
            'units2ndWithoutEvaluations': int(request.form['units2ndWithoutEvaluations']),
            'unemploymentRate': float(request.form['unemploymentRate']),
            'inflationRate': float(request.form['inflationRate']),
            'gdp': float(request.form['gdp'])
        }
        # Convierte el diccionario a DataFrame para facilitar el preprocesamiento
        input_df = pd.DataFrame([input_dict])

        # Preprocesa los datos antes de predecir
        #processed_input = preprocessor.transform(input_df)

        # Realizar la predicción
        prediction = best_model.predict(input_df)

        # Determinar el resultado de la predicción
        result = "Graduado" if prediction[0] == "Graduate" else "Renunciado" if prediction[0] == "Dropout" else "Inscrito"

        # Renderizar una plantilla para mostrar el resultado
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

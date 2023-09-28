from flask import Blueprint, Flask, render_template, request
import numpy as np
from base64 import b64encode
from app.models.KneeMRIClassifier.Model import KneeMRIClassifier

from app.models.KneeXRayClassifier.Model import KneeXRayClassifier

import os

app = Flask(__name__)
kneeXRModel = KneeXRayClassifier()
kneeMRIModel = KneeMRIClassifier()

def BinaryModelLogic(model, img):
    pred = model.forward(img)

    pred = pred[0][0] * 100
    pred_array = np.array([[1, pred], [0, 100 - pred]])
    sorted_pred_array = pred_array[pred_array[:, 1].argsort()][::-1]
    

    dic = {i:[model.class_names[int(x)], f'{y:.2f}'] for i, (x, y) in enumerate(sorted_pred_array.tolist())}

    return dic

@app.route("/")
def main():
    return "Servidor Rodando"

@app.route("/kneeRXClassifier", methods=['GET','POST'])
def classification_apiRX():
    uploaded_file = request.files.get('uploaded_file')
    if uploaded_file:
        file_data = uploaded_file.read() 
        resultado = BinaryModelLogic(model=kneeXRModel, img=file_data)
        return {"doenca": resultado[0][0]}

@app.route("/kneeMRIClassifier", methods=['GET','POST'])
def classification_apiMRI():
    uploaded_file = request.files.get('uploaded_file')
    if uploaded_file:
        file_data = uploaded_file.read() 
        resultado = BinaryModelLogic(model=kneeMRIModel, img=file_data)
        return {"doenca": resultado[0][0]}

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
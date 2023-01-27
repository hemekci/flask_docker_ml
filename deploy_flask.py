
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())


def makePrediction(model, encoder, *flower_attributes):
    prediction_raw = model.predict(flower_attributes)
    prediction_real = encoder.inverse_transform(prediction_raw)
    return prediction_real[0]


@app.route('/predict', methods = ["POST", "GET"])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    flower_attributes = [sepal_length,sepal_width,petal_length,petal_width]
    
    encoder_loaded = joblib.load("saved_models/0_iris_label_encoder.pkl")
    classifier = request.form['classifiers']
    print(classifier)
    classifier_loaded = joblib.load(f"saved_models/{classifier}.pkl")
    
    result = makePrediction(classifier_loaded,encoder_loaded, flower_attributes)
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
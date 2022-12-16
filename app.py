from flask import Flask
from flask import request
from flask_cors import CORS
from prophet.serialize import model_from_json

app = Flask(__name__)
CORS(app)

with open('serialized_model.json', 'r') as fin:
    my_model = model_from_json(fin.read())  # Load model

@app.route("/predict", methods=["GET"])
def predict():
    scores = int(request.args.get('days'))
    future_dates = my_model.make_future_dataframe(periods=7)
    forecast = my_model.predict(future_dates)
    fore_dict = forecast.tail(scores).to_dict('records')
    response = []
    for fore in fore_dict:
        response.append({key: fore[key]
                         for key in fore if key in ["ds", "trend", "yhat"]})
    return response, 200


if __name__ == "__main__":
    app.run()

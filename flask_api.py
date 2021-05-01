from flask import Flask, request, jsonify
import joblib

kmeans = joblib.load('knn-model.pkl')


app = Flask(__name__)


@app.route("/")
def index():
    return "try using kmean endpoint and provide something to classify"


@app.route("/classify", methods=['POST'])
def classify():
    data = request.json
    parameters = data["params"]
    y = kmeans.predict([parameters])
    dict_class = {"classification": str(y[0])}
    return jsonify(dict_class)


if __name__ == '__main__':
    app.run(debug=False)

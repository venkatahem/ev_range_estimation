import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf

app = Flask(__name__)
model = pickle.load(
    open(
        "C:/Users/dkvhe/OneDrive/Documents/vs_code/Projects/CIP/EV range estimation/range_estimation/Deploy-model/range_model_num.pkl",
        "rb",
    )
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [float(x) for x in request.form.values()]
    features = [str(x) for x in request.form.values()]
    print(features)
    print(int_features)
    final_features = [np.array(int_features)]
    input_tensor = tf.reshape(final_features, shape=(1, 12))
    prediction = model.predict(input_tensor)

    # output = round(prediction[0], 2)
    output = prediction

    return render_template(
        "index.html",
        prediction_text="The remaining range should be (KM) {}".format(output),
    )


@app.route("/predict_api", methods=["POST"])
def predict_api():
    """
    For direct API calls trought request
    """
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)

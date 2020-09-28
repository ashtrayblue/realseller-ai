from flask import Flask, request, Response, jsonify
import pickle
import numpy as np

app = Flask(__name__)
app.config.update(DEBUG=True)

@app.route("/")
def hello():
    return "Hello from Python!"

@app.route("/models/prices/prediction", methods=["POST"])
def pred_price():
    app.logger.info("++ predict prices ++")
    app.logger.debug(request.get_json())
    params = request.get_json()
    app.logger.debug(params[3])
    encoder = pickle.load(open('/app/models/model_label_encoder.bin', 'rb'))
    transed_model = encoder.transform([params[3]])
    app.logger.debug(transed_model[0])
    input = params
    input[3] = transed_model[0]
    model = pickle.load(open('/app/models/price_pred_model.bin', 'rb'))
    app.logger.debug(model)
    pred_price = model.predict([input])
    app.logger.debug(pred_price)
    ori_pred = np.expm1(pred_price)

    return jsonify(price=ori_pred[0])

@app.route("/models/frauds/prediction", methods=["POST"])
def pred_frauds():
    app.logger.info("++ predict fraud ++")
    app.logger.debug(request.get_json())
    params = request.get_json()
    model = pickle.load(open('/app/models/fraud_pred_model.bin', 'rb'))
    input = params
    pred = model.predict([input])
    pred_prob = model.predict_proba([input])
    app.logger.debug(pred)
    app.logger.debug(pred_prob)
    
    return jsonify(fraud_prob=pred_prob[0][1])

if __name__ == "__main__":
    app.run(host='0.0.0.0')

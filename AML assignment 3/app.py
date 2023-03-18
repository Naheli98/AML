from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

model = joblib.load('model.joblib')

@app.route('/score', methods=['POST'])
def score_text():
    # Get the text input from the request body
    text = request.json['text']
    
    # Use the score function to get the prediction and propensity
    prediction, propensity = score(text, model, threshold=0.5)
    
    # Create a response dictionary with the prediction and propensity
    response = {'prediction': prediction, 'propensity': propensity}
    
    # Return the response in JSON format
    return jsonify(response)

if __name__ == '__main__':
    app.run()

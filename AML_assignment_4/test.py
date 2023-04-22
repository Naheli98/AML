import warnings
warnings.filterwarnings("ignore")
import os
import requests
from joblib import load
from score import score

model = load('model.joblib')

def test_score_smoke():
    # Smoke test to ensure the function runs without errors
    text = 'This is a test'
    threshold = 0.5
    # Load the trained model from the model.joblib file
    prediction, accuracy = score(text, model, threshold)
    print(prediction, accuracy)

def test_score_format():
    # Format test to ensure the input/output types and formats match the function signature
    output = score("This is a test", model, 0.5)
    assert isinstance(output, tuple)
    assert isinstance(output[0], int)
    assert isinstance(output[1], float)
    print("Output of score():", output)


def test_score_prediction_value():
    # Test to ensure that the prediction value is always either 0 or 1
    prediction, _ = score("This is a test", model, 0.5)
    assert prediction in [0, 1], f"Prediction value is {prediction}, expected 0 or 1"
    print("Test passed: prediction value is either 0 or 1")


def test_score_propensity_value():
    # Test to ensure that the propensity score is always between 0 and 1
    prediction, _ = score("This is a test", model, 0.5)
    assert 0 <= prediction <= 1
    print("Test passed: propensity score is between 0 and 1")

def test_score_threshold_zero():
    # Test to check that when the threshold is set to 0, the prediction is always 1
    prediction, _ = score("This is a test", model, 0)
    assert prediction == 1
    print("Test passed: prediction is 1 when threshold is 0")

def test_score_threshold_one():
    # Test to check that when the threshold is set to 1, the prediction is always 0
    prediction, _ = score("This is a test", model, 1)
    assert prediction == 0
    print("Test passed: prediction is 0 when threshold is 1")

def test_score_obvious_spam():
    # Test to check that an obvious spam input text produces a prediction of 1
    prediction, _ = score("Buy cheap Product now!", model, 0.5)
    assert prediction == 1
    print("Test passed: prediction is 1 for obvious spam input")

def test_score_obvious_non_spam():
    # Test to check that an obvious non-spam input text produces a prediction of 0
    prediction, _ = score("The weather is nice today.", model, 1)
    assert prediction == 0
    print("Test passed: prediction is 0 for obvious non-spam input")
    
def test_flask_integration():
    # Integration test to check the Flask endpoint
    os.system("python app.py &")
    response = requests.post("http://localhost:5000/score", json={"text": "This is a test"})
    data = response.json()
    assert "prediction" in data
    assert "propensity" in data
    assert 0 <= data["prediction"] <= 1
    assert 0 <= data["propensity"] <= 1
    os.system("pkill -f app.py")

def test_docker():
    # Build the Docker image
    os.system("docker build -t my_image .")

    # Launch the Docker container
    os.system("docker run -d -p 8000:8000 --name my_image")

    # Send a request to the local endpoint
    sample_text = "This is a test."
    response = requests.post("http://localhost:8000/score", json={"text": sample_text})

    # Check if the response is as expected
    expected_response = {"result": "success", "prediction": "positive"}
    assert response.status_code == 200
    assert response.json() == expected_response

    # Close the Docker container
    os.system("docker stop my_container")
    os.system("docker rm my_container")
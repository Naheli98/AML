import numpy as np
from sklearn.base import BaseEstimator
from typing import Tuple

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[int, float]:
    # Compute the propensity score
    X = np.array([text])
    propensity = model.predict_proba(X)[0, 1]
    # Compute the prediction
    prediction = int(propensity > threshold)
    return prediction, propensity
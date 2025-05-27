import sys, os
from tensorflow.keras.models import model_from_json

def load_saved_model(json_path, weights_path):
    with open(json_path, 'r') as f:
        model = model_from_json(f.read())  # load the model
    model.load_weights(weights_path)
    return model

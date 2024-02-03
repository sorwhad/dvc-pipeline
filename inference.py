import mlflow
from mlflow.pytorch import load_model
import torch
import torch.nn as nn

import cv2
import numpy as np


mlflow.set_tracking_uri('http://localhost:5005')


model = load_model('models:/LeNet/Production')

softmax = nn.Softmax(dim=1)


def preprocess(img):

    img_resized = cv2.resize(img, (32, 32))
    output = np.transpose(img_resized, (2, 0, 1))

    return torch.FloatTensor(output).unsqueeze(0)


with torch.no_grad():
    img = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = preprocess(img)

    logits = model(tensor)

    print(softmax(logits))



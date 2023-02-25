from torchvision.models import resnet34
import torch.nn as nn
import torch
import streamlit as st

from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import numpy as np

from torchvision.transforms import Compose, ToTensor, Resize

device = torch.device("mps")
transform = Compose([ToTensor(), Resize([150, 150])])

@st.cache
def load_model(path):
    print("Loading model...")
    model = resnet34()
    last_layer = model.fc
    model.fc = nn.Linear(last_layer.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def predict(model, image):
    model.eval()

    transform = Compose([ToTensor(), Resize([150, 150])])

    image = transform(image)
    image = image.reshape([1, 3, 150, 150])
    output = model(image.to(device))
    _, predicted = torch.max(output.data, 1)
    return predicted


def compute_saliency_maps(X, y, model):
    model.eval()
    X = transform(X).reshape([1, 3, 150, 150])
    X.requires_grad_()

    saliency = None

    loss_function = nn.CrossEntropyLoss()
    output = model(X.to(device))
    loss = loss_function(output, y)
    loss.backward()

    saliency, _ = torch.max(torch.abs(X._grad), axis=1)
    return saliency.detach().numpy().reshape([150, 150])

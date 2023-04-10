from torchvision.models import resnet34
import torch.nn as nn
import torch
import streamlit as st
from torchvision.transforms import transforms
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import numpy as np

from torchvision.transforms import Compose, ToTensor, Resize

device = torch.device("mps")
transform = Compose([ToTensor(), Resize([150, 150])])

f = transforms.Compose([
        # transforms.CenterCrop(300),
        transforms.RandomAffine(degrees=100, translate=(0.15, 0.15)),
        transforms.RandomGrayscale(p=0.2),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03),
        transforms.Resize(150)
    ])

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

def smoothing_loss(X):
    loss = 0.0
    loss += torch.sqrt(torch.mean(torch.pow(X[:, :, :, :-1]-X[:, :, :, 1:], 2)))
    loss += torch.sqrt(torch.mean(torch.pow(X[:, :, :, 1:]-X[:, :, :, :-1], 2)))
    loss += torch.sqrt(torch.mean(torch.pow(X[:, :, :-1, :]-X[:, :, 1:, :], 2)))
    loss += torch.sqrt(torch.mean(torch.pow(X[:, :, 1:, :]-X[:, :, :-1, :], 2)))
    return loss


def my_loss(output, y):
    return torch.sum(-1 / 10 * output[:, y])  # + torch.sum(output[:, 1-y])


def generate_images(X, y, model, lr, n):
    model.eval()

    X.requires_grad_()
    X_f = torch.stack([f(x) for x in X for y in range(n)])  # bs*n 150 150 3

    # loss_function = nn.CrossEntropyLoss()
    loss_function = my_loss
    # outputs = torch.stack([model(x.to(device) for x in X_f)]) # bs*n 2
    outputs = model(X_f.to(device))  # bs*n 2

    y_f = torch.stack([y_i for y_i in y for _ in range(n)])  # bs*n

    loss_main = loss_function(outputs, y_f) / n

    smoothing_loss_ = smoothing_loss(X)
    loss = loss_main + smoothing_loss_

    # y.shape: bs

    # bs*n 150 150 3

    loss.backward()
    # if randint(0, 20) == 20:
    X.requires_grad_(False)
    with torch.no_grad():
        X_new = X - lr * X.grad
    X.grad.zero_()

    difference = torch.sum(torch.abs(X_new - X))
    out_of_bound = torch.sum((X_new > 1) + (X_new < 0))

    print(
        loss_main.item(),
        smoothing_loss_.item(),
        difference,
        out_of_bound,
        # output.cpu().detach().numpy().tolist()
    )

    X_new[X_new < 0] = 0
    X_new[X_new > 1] = 1

    return X_new


def generate_image(x, y, model, lr, n):
    X_new = generate_images(x.unsqueeze(0), y, model, lr, n)
    return X_new[0]

def tweak_image(X, y, model):

    im = transform(X)

    n = 16
    lr = 0.3

    new_im = generate_image(im, y, model, lr, n)

    my_bar = st.progress(0)

    for i in range(80):
        my_bar.progress(i/80)
        new_im = generate_image(new_im, y, model, lr, n)

    new_numpy_im = new_im.detach().numpy().transpose(1, 2, 0)

    new_numpy_im[new_numpy_im < 0] = 0
    new_numpy_im[new_numpy_im > 1] = 1

    return new_numpy_im

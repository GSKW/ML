import streamlit as st
import network as net
from PIL import Image
import numpy as np
import PIL
import torch

st.title("Are you a dog or a cat?")

model = net.load_model('../NET.pth')



def heat_to_image(heatmap: PIL.Image, image: PIL.Image) -> PIL.Image:
    heatmap = np.array(heatmap)
    n = 255/np.max(heatmap)
    image = np.array(image)
    heatmap = (heatmap.astype('float64') * n).astype('uint8')
    # print(np.max(image[:, :, 0]))
    # print(heatmap.shape)
    # print(image.shape)
    image[:, :, 0] = np.maximum(heatmap, image[:, :, 0])
    # print(np.max(image[:, :, 0]))
    print(np.max(heatmap))
    return Image.fromarray((image), mode="RGB")


image = st.camera_input(label='Smile :)')
if image is not None:
    img = Image.open(image).convert("RGB")
    pred = net.predict(model, img)
    heatmap = net.compute_saliency_maps(img, pred, model)
    image_heatmap = Image.fromarray((heatmap*255).astype('uint8'), mode="L")
    st.image(heat_to_image(image_heatmap.resize(img.size), img))

    np_tweaked_img = net.tweak_image(img, torch.tensor(1-pred), model)
    tweaked_img = Image.fromarray((np_tweaked_img*255).astype('uint8'), mode='RGB')
    st.image(tweaked_img.resize(img.size))
    if pred.item():
        st.write("U r a dog!!!")
    else:
        st.write("U r a cat!!!")

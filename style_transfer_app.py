import streamlit as st
import cv2
import os
import torch
import numpy as np
import PIL.Image as Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)
vgg.to(device)

def load_img(img_path, img_size = 512, shape = None):
    img = Image.open(img_path).convert('RGB')
    if max(img.size) < img_size:
        img_size = max(img.size)
    if shape is not None:
        img_size = shape
    transform = transforms.Compose([transforms.Resize(img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(img).unsqueeze(0)

def np_convert(tensor):
    img = tensor.cpu().clone().detach().numpy()
    img = img.squeeze()
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    return img.clip(0, 1)

def get_features(img, model):
    layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    for name, layer in model._modules.items():
        img = layer(img)
        if name in layers:
          features[layers[name]] = img
            
    return features

def gram_matrix(tensor):
  _, c, h, w = tensor.size()
  tensor = tensor.view(c, h*w)
  gram= torch.mm(tensor, tensor.t())

  return gram

def style_transfer(content, style):
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    style_weights = {'conv1_1': 1.0, 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

    content_weight = 1
    style_weight = 1e6

    target = content.clone().requires_grad_(True).to(device)

    verbose = 300
    optimizer = optim.Adam([target], lr=0.003)
    num_steps = 12000

    height, width, channels = np_convert(target).shape
    img_array = np.empty(shape=(300, height, width, channels))
    cap_frame = num_steps/300

    ctr = 0

    for i in range(1, num_steps+1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean(torch.square(target_features['conv4_2'] - content_features['conv4_2']))
        style_loss = 0.0

        for layer in style_weights:
            target_feature = target_features[layer]
            target_gm = gram_matrix(target_feature)
            style_gm = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean(torch.square(target_gm - style_gm))
            _, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_loss * content_weight + style_loss * style_weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % verbose == 0:
            print(f'Step {i}, Total Loss: {total_loss.item()}')

            plt.imshow(np_convert(target))
            plt.axis('off')
            plt.show()

        if i % cap_frame == 0:
            img_array[ctr] = np_convert(target)
            ctr += 1

    return target

# Streamlit UI
st.title("Style Transfer App")

# Upload images
primary_image = st.file_uploader("Upload Primary Photo", type=["jpg", "jpeg", "png"])
target_image = st.file_uploader("Upload Target Photo", type=["jpg", "jpeg", "png"])

content = load_img(primary_image, shape=(512, 512)).to(device)
style = load_img(target_image, shape=(512, 512)).to(device)

"""
Upload your Photos and Get result soon... real soon... Good Night ^_^
"""

# Style transfer button
if st.button("Perform Style Transfer"):
    if primary_image is not None and target_image is not None:
        # Read and preprocess images
        primary_image = Image.open(primary_image).convert("RGB")
        target_image = Image.open(target_image).convert("RGB")

        primary_image = np.array(primary_image)
        target_image = np.array(target_image)

        # Perform style transfer
        stylized_image = style_transfer(primary_image, target_image)

        # Display the images
        st.image([primary_image, target_image, stylized_image.numpy()], caption=['Primary', 'Target', 'Stylized'], width=300)

    else:
        st.warning("Please upload both primary and target photos.")

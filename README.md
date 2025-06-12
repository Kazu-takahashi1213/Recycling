# Recycling AI - Smart Waste Classification for Paderborn

<br>

A deep learning–powered app that helps Paderborn residents determine how to sort their household waste—instantly.

> Because garbage segregation in Germany can be confusing, especially for newcomers, this app provides a fast and accurate classification based on an image.

<br>

---

<br>

## Overview

<br>

This project uses computer vision to classify images of household waste into **five categories** defined by Paderborn’s waste policy.

<br>

Built using:
- Transfer learning with MobileNetV2
- A Kaggle garbage dataset (12 categories, 15,000+ images)
- Streamlit app for interactive use
- Evaluation with confusion matrix

<br>

---

<br>

## Background: Waste Categories in Paderborn

<br>

In Paderborn, garbage is sorted into **five main bins**:

<br>

| Category | Description | Examples |
|----------|-------------|----------|
| Restmüll (Residual) | Non-recyclable, contaminated waste | Dirty packaging, textiles |
| Bioabfall (Biowaste) | Organic waste | Vegetable peels, food scraps |
| Altpapier (Paper) | Clean paper and cardboard | Newspapers, boxes |
| Wertstofftonne (Recyclables) | Plastic/metal recyclables | PET bottles, cans |
| Altglas (Glass) | Glass containers | Bottles, jars |

<br>

---

<br>

## Dataset

<br>

Originally, I tried collecting 100 images per category using `BingImageCrawler`, but this approach lacked quality and quantity.

<br>

 So instead, I used [Kaggle: Garbage Classification Dataset](https://www.kaggle.com/datasets) with:

<br>

- 15,150 images
- 12 original classes:
  - paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, trash

<br>

Then I **mapped the 12 categories → 5 categories** to match Paderborn’s policy:

<br>

| Paderborn Category | Mapped Classes |
|-------------------|----------------|
| `biowaste`        | biological     |
| `glass`           | glass, green-glass, white-glass |
| `paper`           | paper, cardboard |
| `wertstoff`       | metal, plastic |
| `residual`        | trash, shoes, clothes |

<br>

> Batteries were excluded from the model.

<br>

---

<br>

## Model Training: `train_model_kaggle_mapped.py`

<br>

### Key Settings

<br>

- Input size: **224 × 224**
- Batch size: **32**
- Epochs: **10**
- Validation split: **20%**
- Image augmentation: rotation, zoom, flipping, shifting
- Preprocessing: `rescale = 1./255`

<br>

### Model Architecture

<br>

Using **MobileNetV2** with transfer learning:

<br>

```text
Input (224x224x3)
↓
MobileNetV2 (pre-trained on ImageNet, frozen)
↓
GlobalAveragePooling2D
↓
Dropout(0.3)
↓
Dense(num_classes, softmax)

```

## About MobileNetV2

<br>

MobileNetV2 is a lightweight and efficient convolutional neural network architecture designed for mobile and embedded devices.

Why use MobileNetV2?

<br>

Pre-trained on ImageNet → strong general features

<br>

Low computational cost → faster training and inference

<br>

Works well with limited datasets

<br>

Ideal for transfer learning

<br>

## Key Architecture Features

<br>

Depthwise Separable Convolutions: drastically reduce parameters

<br>

Inverted Residuals: skip connections that pass through bottlenecks

<br>

Linear Bottlenecks: avoid nonlinearities where unnecessary

<br>

## In our project, we freeze the MobileNetV2 base (weights are not updated) and use it as a feature extractor, adding our own classification head on top.

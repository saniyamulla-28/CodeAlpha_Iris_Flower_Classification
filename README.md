# CodeAlpha Task 1 — Iris Flower Classification

## Objective

The objective of this project is to classify Iris flowers into three species:

- Setosa
- Versicolor
- Virginica

The classification is performed using flower measurements such as sepal
length, sepal width, petal length, and petal width.

## Dataset

The Iris dataset is obtained using Scikit-learn's built-in Iris dataset.

The dataset contains:

- 150 flower samples
- 4 input features
- 3 target classes

## Features

The model uses the following measurements:

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

## Machine Learning Model

The project uses:

- StandardScaler for feature scaling
- Logistic Regression for classification

## Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

## Model Performance

The model achieved:

- Test Accuracy: 93.33%
- 5-Fold Cross Validation Accuracy: 96.00%

## Project Structure

```text
CodeAlpha_Iris_Flower_Classification/
│
├── data/
│   └── iris.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   └── feature_scatter.png
│
├── iris_classification.py
├── README.md
└── requirements.txt
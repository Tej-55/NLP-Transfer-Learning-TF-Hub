# NLP Transfer Learning with TF-Hub

This repository contains a Python notebook that demonstrates how to perform NLP transfer learning using TensorFlow Hub. The notebook provides step-by-step instructions on how to train various text classification models using pre-trained embeddings from TensorFlow Hub.

## Table of Contents

- [Introduction](#introduction)
- [Importing the Dataset](#importing-the-dataset)
- [Compiling Models](#compiling-models)
- [Training and Evaluating Models](#training-and-evaluating-models)
- [Comparing Accuracy and Loss Curves](#comparing-accuracy-and-loss-curves)
- [Visualizing Metrics with TensorBoard](#visualizing-metrics-with-tensorboard)

## Introduction

The notebook starts by importing the necessary libraries, including TensorFlow, TensorFlow Hub, and other required packages. It then proceeds to import the dataset, which is the Quora Insincere Questions Classification data. The dataset is read into a pandas DataFrame and explored to understand its structure.

Next, the notebook compiles different text classification models using pre-trained embeddings from TensorFlow Hub. The models are trained and evaluated using the dataset, and the accuracy and loss curves are plotted for comparison.

Finally, the notebook demonstrates how to visualize the training metrics using TensorBoard. The TensorBoard logs are saved in a temporary directory, and TensorBoard is used to display the accuracy and loss curves over the training epochs.


## Importing the Dataset

The notebook uses the Quora Insincere Questions Classification dataset. The dataset can be downloaded from [this link](https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip). The notebook provides code to download and read the dataset into a pandas DataFrame.


## Compiling Models

The notebook defines a function to compile text classification models. The function takes a module URL from TensorFlow Hub, the embedding size, a name for the model, and a flag indicating whether the embedding layer should be trainable. It creates a sequential model with a hub layer for the pre-trained embedding, followed by dense layers for classification. The model is compiled with an optimizer, loss function, and metrics.

## Training and Evaluating Models

The notebook trains and evaluates various text classification models using different pre-trained embeddings from TensorFlow Hub. It uses the `train_and_evaluate_model` function to train each model and stores the training history in a dictionary. The function takes the module URL, embedding size, model name, and trainable flag as input. It fits the model to the training data and evaluates it on the validation data.

## Comparing Accuracy and Loss Curves
After training the models, the notebook includes a section to compare the accuracy and loss curves of the trained models. It uses the TensorFlow docs library to plot the curves and provides visual insights into the training progress and performance of each model. The curves help in assessing the convergence and performance of the models over epochs.

## Visualizing Metrics with TensorBoard
The notebook also includes a section on visualizing metrics using TensorBoard. TensorBoard is a powerful visualization toolkit included with TensorFlow that allows for real-time tracking and visualization of various metrics during training. By running the TensorBoard server, you can visualize and analyze the training progress, including accuracy, loss, and other custom metrics. The notebook provides instructions on how to set up and launch TensorBoard to view the metrics.

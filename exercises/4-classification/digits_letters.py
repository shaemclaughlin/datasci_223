# -*- coding: utf-8 -*-
"""digits_letters.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uBOdbie7rkfNP_w7abTTpLt_bj-CoZ3A
"""

# Commented out IPython magic to ensure Python compatibility.
# Install required packages
# %pip install -q numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost emnist
# %reset -f

# Import packages
import os
import string
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emnist
from IPython.display import display, Markdown

# ML packages
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# XGBoost (SVM)
from xgboost import XGBClassifier
# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Constants
SIZE = 28
REBUILD = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper functions

# Convert an integer label to the corresponding uppercase character
def int_to_char(label):
    if label < 10:
      return str(label)
    elif label < 36:
      return chr(label - 10 + ord('A'))
    else:
      return chr(label - 36 + ord('a'))

# Display a single image and its corresponding label
def show_image(row):
  image = row['image']
  label = row['label']
  plt.imshow(image, cmap='gray')
  plt.title('Label: ' + int_to_char(label))
  plt.axis('off')
  plt.show()

# Display a list of images as a grid of num_cols columns
def show_grid(data, title=None, num_cols=5, fig_size=(20,10)):
  num_images = len(data)
  num_rows = (num_images - 1) // num_cols + 1
  fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
  if title is not None:
    fig.subtitle(title, fontsize=16)
  for i in range(num_rows):
      for j in range(num_cols):
          index = i * num_cols + j
          if index < num_images:
              axes[i, j].imshow(data.iloc[index]['image'], cmap='gray')
              axes[i, j].axis('off')
              label = int_to_char(data.iloc[index]['label'])
              axes[i, j].set_title(label)
  plt.show()

# Get a random image of a given label from the dataset
def get_image_by_label(data, label):
  images = data[data['label'] == label]['image'].tolist()
  return random.choice(images)

# Plot the training and validation accuracy during the training of the model
def plot_accuracy(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
  plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

# Plot the training and validation loss during the training of the model
def plot_loss(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

# Display metrics for a model
def display_metrics(task, model_name, metrics_dict):
  metrics_df = pd.DataFrame()
  cm_df = pd.DataFrame()
  for key, value in metrics_dict[task][model_name].items():
    if type(value) == np.ndarray:
      cm_df = pd.DataFrame(value, index=['Actual 0','Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    else:
      metrics_df[key] = [value]
  display(Markdown(f'## Performance Metrics: {model_name}'))
  display(metrics_df)
  display(Markdown(f'## Confusion Matrix: {model_name}'))
  display(cm_df)

# Load EMNIST 'byclass' data

# Extract the training split as images and labels
image, label = emnist.extract_training_samples('byclass')

# Add columns for each pixel value (28x28 = 784 columns)
train = pd.DataFrame()

# Add a column with the image data as a 28x28 array
train['image'] = list(image)
train['image_flat'] = train['image'].apply(lambda x: np.array(x).reshape(-1))

# Add a column showing the label
train['label'] = label

# Create a binary label column for each row, 0 for digits (0-9), 1 for letters (10-61)
#binary_label = np.array([(label >= 10).astype(int)] for l in label)

# Add binary label column
#train['binary_label'] = binary_label

# Convert labels to characters
class_label = np.array([int_to_char(l) for l in label])

# Add a column with the character corresponding to the label
train['class'] = class_label

# Repeat for the test split
image, label = emnist.extract_test_samples('byclass')
#binary_label = np.array([(label >= 10).astype(int)] for l in label)
class_label = np.array([int_to_char(l) for l in label])
test = pd.DataFrame()
test['image'] = list(image)
test['image_flat'] = test['image'].apply(lambda x: np.array(x).reshape(-1))
test['label'] = label
#valid['binary_label'] = binary_label
test['class'] = class_label

# Combine the training and test data for later use
byclass = pd.concat([train, test], ignore_index=True)

# Create a dictionary for performance metrics
metrics_dict = {}
metrics_dict['letter_vs_digit'] = {}
metrics_dict['validation'] = {}

# Classify the images as letters or digits
byclass['is_letter'] = byclass['label'] >= 10
train['is_letter'] = train['label'] >= 10
test['is_letter'] = test['label'] >= 10

# Display the first few rows of the dataset
display(byclass.head())

# Define hyperparameters for GridSearch
logistic_params = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
random_forest_params = {'n_estimators':[10,50,100]}
xgboost_params = {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1]}

# Define models
models = [
    {'name': 'logistic_regression', 'model': LogisticRegression(max_iter=1000, random_state=42), 'params': logistic_params},
    {'name': 'random_forest', 'model': RandomForestClassifier(random_state=42), 'params': random_forest_params},
    {'name': 'xgboost', 'model': XGBClassifier(random_state=42), 'params': xgboost_params},
    {'name': 'neural_network', 'model': Sequential([Flatten(input_shape=(784,)), Dense(1, activation='sigmoid')]), 'params': None}
]

# Define scores for model evaluation
scores = ['accuracy', 'precision', 'recall', 'f1']

#sample_size = 2000
#sample = byclass.sample(sample_size, random_state=42)
#valid_frac = 0.3
#valid = sample.sample(frac=valid_frac, random_state=42)
#train_test = sample.drop(valid.index)



# Split data into train/test and validation sets
valid_frac = 0.3
valid = byclass.sample(frac=valid_frac, random_state=42)
train_test = byclass.drop(valid.index)

# K-fold cross-validation
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

winning_model = None
winning_name = None
wins = 0

for train_index, test_index in kf.split(train_test['image_flat'].apply(lambda x: tuple(x)), train_test['is_letter']):
  train = train_test.iloc[train_index]
  test = train_test.iloc[test_index]

  # Preprocess data
  train_scaled = scaler.fit_transform(np.vstack(train['image_flat'].values))
  test_scaled = scaler.fit_transform(np.vstack(test['image_flat'].values))

  # Split dataset into features and labels
  #x_train_scaled = np.vstack(train_scaled['image_flat'].values)
  y_train = train['is_letter']
  #x_test_scaled = np.vstack(test_scaled['image_flat'].values)
  y_test = test['is_letter']

  for model_info in models:
    model_name = model_info['name']
    model = model_info['model']
    params = model_info['params']

    # Check if model is a neural network and compile it
    if model_name=="neural_network":
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Train and evaluate model
    if params is not None:
      # Perform GridSearch
      grid = GridSearchCV(model, params, cv=5, scoring=make_scorer(accuracy_score))
      grid.fit(train_scaled, y_train)
      model = grid.best_estimator_

    else:
      # Train model normally
      model.fit(train_scaled, y_train)

    y_pred = (model.predict(test_scaled) > 0.5).astype("int32")

    # Calculate and store performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    metrics_dict['letter_vs_digit'][model_name] = \
      {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}

    # Determine if current model is the new winning model (based on f1 score)
    if wins < f1:
      wins = f1
      winning_model = model
      winning_name = model_name

# Apply winning model to validation set
valid_scaled = scaler.transform(np.vstack(valid['image_flat'].values))
y_pred = winning_model.predict(valid_scaled)

# If winning model is a neural network, convert probabilistic outputs to binary
if isinstance(winning_model, Sequential):
  y_pred = (y_pred > 0.5).astype("int32")

# Calculate and display performance metrics for winning model on validation set
acc = accuracy_score(valid['is_letter'], y_pred)
prec = precision_score(valid['is_letter'], y_pred)
rec = recall_score(valid['is_letter'], y_pred)
f1 = f1_score(valid['is_letter'], y_pred)
cm = confusion_matrix(valid['is_letter'], y_pred)

metrics_dict['validation'][winning_name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}

# Display metrics
display(Markdown(f'# Train/Test'))
display_metrics('letter_vs_digit', winning_name, metrics_dict)
display(Markdown(f'# Validation'))
display_metrics('validation', winning_name, metrics_dict)
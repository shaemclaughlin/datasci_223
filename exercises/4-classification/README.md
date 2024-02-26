The script for this assignment can be found here: https://github.com/shaemclaughlin/datasci_223/blob/main/exercises/4-classification/homework4.5.ipynb 

# Classifying Letters 'a' to 'g'

This script builds a machine learning model which aims to classify images of handwritten letters from 'a' to 'g'. These images were sourced from the EMNIST dataset and processed so that each image is a 28x28 array of pixel intensities.

## Sampling

To expedite code testing and debugging, I initially operate on a sample of 1000 observations randomly drawn from our total pool of images.

```python
# Define small sample dataset for testing
sample_size = 1000
a2g_sample = a2g.sample(sample_size, random_state=42)
```

## Data Preprocessing

The features (handwritten images) and labels ('a' to 'g') are separated to prepare the data for model fitting. The data is split into a training set (80%) to train the model and a test set (20%) to validate the model's performance. The feature data is standardized (mean = 0, standard deviation = 1) for optimal model performance.

```python
# Define features and labels
X = np.vstack(a2g_sample['image_flat'].values)
y = a2g_sample['class']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Model Training and Hyper-parameter Tuning

For this task, I used a Random Forest classifier as my chosen model. Random forests operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees. 

I used `GridSearchCV` to tune the `n_estimators` hyper-parameter in the Random Forest algorithm. This represents the number of trees in the forest. Typically, the larger the number of trees, the better the performance of the model.

After identifying the best hyper-parameter with cross-validation on the training set, I re-trained the model using this configuration.

```python
# Define model
model = RandomForestClassifier(random_state=42)
params = {'n_estimators': [100, 500, 1000]}

# Perform GridSearch to find the best params
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train_scaled, y_train)

print("Best Hyperparameters::\n{}".format(grid.best_params_))

# Train the model using the best params
best_rf = RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'], random_state=42)
best_rf.fit(X_train_scaled, y_train)
```

## Model Evaluation

I applied the fitted model to the evaluation dataset, generating predictions. These are compared to the actual values to evaluate model performance. I compute the accuracy score, the confusion matrix, and the classification report providing overall and per-class precision, recall, and f-1 score metrics.

```python
# Predict test set results
y_pred = best_rf.predict(X_test_scaled)

# Evaluate and store results
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
metrics_dict['a_to_g']={'accuracy': accuracy, 'classification_report':report, 'confusion_matrix': cm}

# Display metrics
display(Markdown(f'### Accuracy:'))
print(metrics_dict['a_to_g']['accuracy'])

display(Markdown(f'### Classification report:'))
print(metrics_dict['a_to_g']['classification_report'])

display(Markdown(f'### Confusion Matrix:'))
print(metrics_dict['a_to_g']['confusion_matrix'])
```
## Observations

From the classification report, all classes have a high recall score which is promising. However, the 'c' class demonstrates lower precision indicating this class is often misclassified. I was able to improve the performance of 'c' by increasing the size of the dataset.

# Classifying Letters as Uppercase/Lowercase

This script builds a machine learning model that differentiates between uppercase and lowercase letters. A binary class variable ('is_upper') is created to represent lowercase and uppercase letters.

## Data Preparation

The input letter images are grouped based on whether the corresponding letter is uppercase or lowercase. As such, aside from the pixel array content, a binary 'is_upper' class is also provided as input. 

```python
# Create an 'is_upper' column and set the values
abcxyz = abcxyz.copy()
abcxyz['is_upper'] = abcxyz['class'].apply(lambda x: x.isupper())
```

## Preprocessing and Model Training

The data is partitioned into a training and validation set, using an 80-20 split. In addition, the pixel values of each image are standardized. 

Four different models are chosen for comparison: Logistic Regression, Random Forest, XGBoost, and a simple Neural Network. For each model, hyperparameters are tuned to optimize performance and the model is then trained on the training data subset. Lastly, model performance is evaluated based on its F1-score.

```python
# Define hyperparameters
logistic_params = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
random_forest_params = {'n_estimators':[10,50,100]}
xgboost_params = {'n_estimators': [10,50,100], 'learning_rate': [0.01, 0.1]}

# Define models
models = [
    {'name': 'logistic_regression', 'model': LogisticRegression(max_iter=1000, random_state=42), 'params': logistic_params},
    {'name': 'random_forest', 'model': RandomForestClassifier(random_state=42), 'params': random_forest_params},
    {'name': 'xgboost', 'model': XGBClassifier(random_state=42), 'params': xgboost_params},
    {'name': 'neural_network', 'model': Sequential([Flatten(input_shape=(784,)), Dense(1, activation='sigmoid')]), 'params': None}  
]
```

## Evaluation

Once the best performing model has been determined, it is evaluated against a separate validation set. The Confusion Matrix, Accuracy, Precision, Recall, and F1 scores are produced for this model. Additionally, results are stored in a dictionary for potential later use.

```python
# Calculate and display performance metrics for winning model on validation set
acc = accuracy_score(valid['is_upper'], y_pred)
prec = precision_score(valid['is_upper'], y_pred)
rec = recall_score(valid['is_upper'], y_pred)
f1 = f1_score(valid['is_upper'], y_pred)
cm = confusion_matrix(valid['is_upper'], y_pred)

metrics_dict['validation'][winning_name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}

# Display metrics
display(Markdown(f'# Train/Test'))
display_metrics('upper_vs_lower', winning_name, metrics_dict)
display(Markdown(f'# Validation'))
display_metrics('validation', winning_name, metrics_dict)
```

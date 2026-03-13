"""
Trains and evaluates a simple Multi-Layer Perceptron (MLP)
on the Reuters newswire topic classification dataset.

This version integrates MLflow for experiment tracking using:
1. Autologging (automatic logging of model artifacts and metrics)
2. Manual logging for selected parameters and evaluation metrics
"""

from __future__ import print_function

# ------------------------------------------------
# Imports
# ------------------------------------------------
import numpy as np

# TensorFlow / Keras
from tensorflow import keras
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.preprocessing.text import Tokenizer

# MLflow
import mlflow
import mlflow.tensorflow


# ------------------------------------------------
# Hyperparameters
# ------------------------------------------------
max_words = 2000
batch_size = 32
epochs = 5
learning_rate = 0.001


# ------------------------------------------------
# Enable MLflow Autologging
# ------------------------------------------------
# Automatically logs:
# - model architecture
# - optimizer parameters
# - training metrics
# - TensorFlow model artifacts
mlflow.tensorflow.autolog()


# ------------------------------------------------
# Start an MLflow Run
# ------------------------------------------------
with mlflow.start_run():

    # Tag the run for easy experiment identification
    mlflow.set_tag("project", "reuters_classification")

    # ------------------------------------------------
    # Load Dataset
    # ------------------------------------------------
    print('Loading data...')

    (x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=max_words,
        test_split=0.2
    )

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    # ------------------------------------------------
    # Log Key Training Parameters (Manual Logging)
    # ------------------------------------------------
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)

    # ------------------------------------------------
    # Determine Number of Output Classes
    # ------------------------------------------------
    num_classes = int(np.max(y_train) + 1)
    print(num_classes, 'classes')

    # ------------------------------------------------
    # Vectorize Text Data
    # ------------------------------------------------
    print('Vectorizing sequence data...')

    tokenizer = Tokenizer(num_words=max_words)

    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # ------------------------------------------------
    # One-Hot Encode Labels
    # ------------------------------------------------
    print('Convert class vector to binary class matrix')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    # ------------------------------------------------
    # Build Neural Network Model
    # ------------------------------------------------
    print('Building model...')

    model = Sequential()

    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # ------------------------------------------------
    # Compile Model
    # ------------------------------------------------
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # ------------------------------------------------
    # Train Model
    # ------------------------------------------------
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1
    )

    # ------------------------------------------------
    # Evaluate Model
    # ------------------------------------------------
    score = model.evaluate(
        x_test,
        y_test,
        batch_size=batch_size,
        verbose=1
    )

    test_loss = score[0]
    test_accuracy = score[1]

    print('Test score:', test_loss)
    print('Test accuracy:', test_accuracy)

    # ------------------------------------------------
    # Manual Metric Logging
    # ------------------------------------------------
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
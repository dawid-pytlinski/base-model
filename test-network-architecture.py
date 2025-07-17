
import pytest
import numpy as np
from keras import models, layers
from keras.utils import to_categorical
from keras.datasets import mnist

@pytest.fixture
def prepare_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels

def test_input_shape(prepare_data):
    train_images, _, _, _ = prepare_data
    assert train_images.shape == (60000, 784), "Train images must be 60000x784"

def test_label_encoding(prepare_data):
    _, train_labels, _, _ = prepare_data
    assert train_labels.shape == (60000, 10), "Labels must be one-hot encoded"

def test_model_output_shape(prepare_data):
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))

    sample_input = np.random.rand(1, 28*28).astype('float32')
    output = model.predict(sample_input)
    assert output.shape == (1, 10), "Model output shape must be (1, 10)"

def test_training_runs(prepare_data):
    train_images, train_labels, _, _ = prepare_data
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images[:1000], train_labels[:1000],
                        epochs=1, batch_size=128, verbose=0)
    assert 'loss' in history.history, "Training did not produce loss"

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, ))) # layers -> jako filtr danych oraz Dese, jako gęstość sieci. 
network.add(layers.Dense(10, activation='softmax')) #Funkcja Softmax zawsze na ostatniej warstwie sieci neuronowej. 

network.compile(optimizer='rmsprop', # lub 'adam'.
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape(10000, 28 * 28)
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print("Test acc: ", test_acc)
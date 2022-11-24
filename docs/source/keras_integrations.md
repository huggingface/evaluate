# Working with Keras and Tensorflow



Evaluate can be easily intergrated into your Keras and Tensorflow workflow. We'll demonstrate two ways of incorporating Evaluate into model training, using the Fashion MNIST example dataset. We'll train a standard classifier to predict two classes from this dataset, and show how to use a metric as a callback during training or after for evaluation. 


```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import evaluate

# We pull example code from Keras.io's guide on classifying with MNIST
# Located here: https://keras.io/examples/vision/mnist_convnet/

# Model / data parameters
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


# Only select tshirts/tops and trousers, classes 0 and 1
def get_tshirts_tops_and_trouser(x_vals, y_vals):
    mask = np.where((y_vals == 0) | (y_vals == 1))
    return x_vals[mask], y_vals[mask]

x_train, y_train = get_tshirts_tops_and_trouser(x_train, y_train)
x_test, y_test = get_tshirts_tops_and_trouser(x_test, y_test)


# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)
```

# Callbacks

Suppose we want to keep track of model metrics while a model is training. We can use a Callback in order to calculate this metric during training, after an epoch ends. 

We'll define a callback here that will take a metric name and our training data, and have it calculate a metric after the epoch ends. 


```python
class MetricsCallback(keras.callbacks.Callback):

    def __init__(self, metric_name, x_data, y_data) -> None:
        super(MetricsCallback, self).__init__()

        self.x_data = x_data
        self.y_data = y_data
        self.metric_name = metric_name
        self.metric = evaluate.load(metric_name)

    def on_epoch_end(self, epoch, logs=None):
        m = self.model 
        # Ensure we get labels of 1 or 0
        training_preds = np.round(m.predict(self.x_data))
        training_labels = self.y_data

        # Compute score and save
        score = self.metric.compute(predictions = training_preds, references = training_labels)
        
        print(f"At end of epoch {epoch}, {self.metric_name} is {score[self.metric_name]}")
```

After callback creation, we can pass it as such in order to activate it. 


```python
batch_size = 128
epochs = 2

model.compile(loss="binary_crossentropy", optimizer="adam")

model_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, 
callbacks = [MetricsCallback(x_data = x_train, y_data = y_train, metric_name = "accuracy")])
```

## Using an Evaluate Metric for... Evaluation!

We can of course use the same metric outside of model training. Here, we check accuracy of the model after training on our train and test sets. 


```python
acc = evaluate.load("accuracy")

train_preds = np.round(model.predict(x_train))
train_labels = y_train

test_preds = np.round(model.predict(x_test))
test_labels = y_test
```


```python
print("Train accuracy is : ", acc.compute(predictions = train_preds, references = train_labels))
print("Test accuracy is : ", acc.compute(predictions = test_preds, references = test_labels))
```

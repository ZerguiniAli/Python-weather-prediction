import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pydot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import model_to_dot
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('seattle-weather.csv')
print(df)

# Normalize numerical features
scaler = MinMaxScaler()
df[['precipitation', 'temp_max', 'temp_min', 'wind']] = scaler.fit_transform(
    df[['precipitation', 'temp_max', 'temp_min', 'wind']]
)

# One-hot encode weather labels
encoder = OneHotEncoder(sparse=False)
weather_labels = encoder.fit_transform(df[['weather']])
weather_categories = encoder.categories_[0]

df_encoded = pd.concat([df, pd.DataFrame(weather_labels, columns=weather_categories)], axis=1)

# Split the data into train and test sets
X = df_encoded[['precipitation', 'temp_max', 'temp_min', 'wind']].values
y = df_encoded[weather_categories].values

print(weather_categories)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(4,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(weather_categories), activation="softmax")
])

# Compile and train the model
model.compile(loss=keras.losses.CategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Plot the accuracy curve
accuracy = history.history['accuracy']
plt.plot(accuracy)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Save the model architecture diagram as a PNG image
plot_model(model, to_file='ann_diagram.png', show_shapes=True, show_layer_names=True)

# Convert the model architecture to GraphViz format
dot_graph = model_to_dot(model, show_shapes=True, show_layer_names=False)

# Modify the GraphViz graph to use card-style nodes
dot_graph.set_node_defaults(shape='box', style='rounded')

# Save the modified GraphViz graph as a PNG image
(graph,) = pydot.graph_from_dot_data(dot_graph.to_string())
graph.write_png('ann_diagram_with_cards.png')

# Make predictions
new_data = {
    'precipitation': [0.0],
    'temp_max': [31.0],
    'temp_min': [13.8],
    'wind': [2.0]
}

new_df = pd.DataFrame(new_data)
new_df[['precipitation', 'temp_max', 'temp_min', 'wind']] = scaler.transform(new_df)
predictions = model.predict(new_df)
predicted_labels = [weather_categories[prediction.argmax()] for prediction in predictions]
print("Predicted weather:", predicted_labels)


# Save the TensorFlow model
keras_file = "my_model.h5"
keras.models.save_model(model, keras_file)

# Convert the TensorFlow model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("weather.tflite", "wb") as f:
    f.write(tflite_model)

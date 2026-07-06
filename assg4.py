#TASK 1
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Task 1: Data Loading and Preprocessing

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0, 1, 2)

# 2. Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Perform feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Normalize training features
X_test = scaler.transform(X_test)       # Normalize testing features

# 4. Encode the target labels using one-hot encoding
y_train = to_categorical(y_train, num_classes=3)  # Convert to one-hot encoded format
y_test = to_categorical(y_test, num_classes=3)    # Convert to one-hot encoded format

# Print the processed data shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


#TASK 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Task 2: Neural Network Construction

# Build the neural network
model = Sequential([
    Dense(8, input_dim=4, activation='relu'),  # Hidden layer with 8 neurons
    Dense(3, activation='softmax')            # Output layer with 3 neurons for 3 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

#TASK 3
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=5,
                    validation_data=(X_test, y_test))

# Optional: Print training summary
print("Training complete.")

#TASK 4
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

# Print the accuracy
print(f"Test Accuracy: {accuracy * 100:.2f}%")



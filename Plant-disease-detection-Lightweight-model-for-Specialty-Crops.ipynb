# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D, Dense, Input, Concatenate
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set variables and hyperparameters
input_shape = (224, 224, 3)          # Input image size (224x224 with 3 color channels)
num_classes = 16                     # Number of disease classes in the dataset
output_activation = 'softmax'        # Softmax activation function for multi-class classification
activation_function = 'relu'         # ReLU activation function for layers
optimizer = Adam(learning_rate=0.001)# Adam optimizer with learning rate of 0.001
batch_size = 32                      # Batch size for training
epochs = 50                          # Number of epochs for training
loss_function = 'categorical_crossentropy' # Loss function for classification

# Directory paths for training, validation, and test data
train_dir = '/content/drive/MyDrive/Research/Data/Dataset/train'
val_dir = '/content/drive/MyDrive/Research/Data/Dataset/validate'
test_dir = '/content/drive/MyDrive/Research/Data/Dataset/test'

# Data Preprocessing: Load images from directories and apply augmentation for training
def preprocess_data(train_dir, val_dir, test_dir, input_size=(224, 224), batch_size=batch_size):
    datagen = ImageDataGenerator(
        rescale=1./255,               # Normalize pixel values to [0, 1]
        rotation_range=20,            # Randomly rotate images up to 20 degrees
        width_shift_range=0.2,        # Randomly shift images horizontally by up to 20%
        height_shift_range=0.2,       # Randomly shift images vertically by up to 20%
        horizontal_flip=True          # Randomly flip images horizontally
    )

    # Load training, validation, and test data from directories
    train_generator = datagen.flow_from_directory(
        train_dir, target_size=input_size, batch_size=batch_size, class_mode='categorical')
    val_generator = datagen.flow_from_directory(
        val_dir, target_size=input_size, batch_size=batch_size, class_mode='categorical')
    test_generator = datagen.flow_from_directory(
        test_dir, target_size=input_size, batch_size=batch_size, class_mode='categorical')

    return train_generator, val_generator, test_generator

# Model Architecture: Build a lightweight CNN with stacked convolutional layers and global max pooling
def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # First convolutional layer
    conv1 = Conv2D(32, (3, 3), activation=activation_function)(inputs)
    gmpl1 = GlobalMaxPooling2D()(conv1)
    fc1 = Dense(64, activation=activation_function)(gmpl1)

    # Second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation=activation_function)(inputs)
    gmp2 = GlobalMaxPooling2D()(conv2)
    fc2 = Dense(64, activation=activation_function)(gmp2)

    # Third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation=activation_function)(inputs)
    gmp3 = GlobalMaxPooling2D()(conv3)
    fc3 = Dense(64, activation=activation_function)(gmp3)

    # Concatenate feature vectors from all three convolutional layers
    concatenate1 = Concatenate()([fc1, fc2])
    concatenate2 = Concatenate()([concatenate1, fc3])

    # Fully connected layers after concatenation
    fc4 = Dense(128, activation=activation_function)(concatenate2)
    fc5 = Dense(64, activation=activation_function)(fc4)

    # Output layer for multi-class classification
    outputs = Dense(num_classes, activation=output_activation)(fc5)

    # Build the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Compile the model: Configure the optimizer, loss function, and evaluation metrics
def compile_model(model):
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# Train the model: Fit the model to the training data and validate on validation data
def train_model(model, train_generator, val_generator, epochs=epochs):
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)
    return history

# Evaluate the model: Test the model on the test data and return evaluation results
def evaluate_model(model, test_generator):
    results = model.evaluate(test_generator)
    return results

# Execute and Test the model

# Load and preprocess data
train_generator, val_generator, test_generator = preprocess_data(train_dir, val_dir, test_dir)

# Build and compile the model
model = create_model(input_shape, num_classes)
compile_model(model)

# Train the model and save training history
history = train_model(model, train_generator, val_generator)

# Evaluate the model on test data
test_results = evaluate_model(model, test_generator)

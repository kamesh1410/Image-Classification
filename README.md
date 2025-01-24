# 🌟 Fashion MNIST Classification with Neural Networks 🌟

This project demonstrates how to classify Fashion MNIST images using a deep neural network built with TensorFlow and Keras. The dataset consists of 60,000 28x28 grayscale images of 10 fashion categories, such as T-shirts, trousers, shoes, etc. The goal is to train a model to predict the correct category of each image.

---

## 🚀 Project Overview

This project covers the following aspects:
- Loading and visualizing the Fashion MNIST dataset.
- Preprocessing the data (normalization).
- Building a deep neural network for classification.
- Training the model and evaluating its performance.
- Visualizing the predictions and analysis.

---

## 🛠️ Steps to Execute the Project

### 1. 📦 Install Required Libraries

Before running the project, you need to install the following libraries:


---

### 2. 📂 Load and Visualize the Data

The dataset is loaded using `tf.keras.datasets.fashion_mnist`. Key steps include:
- Load the training and test data.
- Visualize random images from the training set.

The images are normalized to values between 0 and 1 by dividing by 255.

---

### 3. 🧠 Build the Neural Network Model

The model architecture includes:
- **Flatten Layer**: Flatten the 28x28 images into 1D vectors.
- **Dense Layers**: Fully connected layers with ReLU activation, L2 regularization, and dropout for regularization.
- **BatchNormalization**: To improve training stability.
- **Output Layer**: Softmax activation for multi-class classification.

---

### 4. ⚡ Compile and Train the Model

The model is compiled using the Adam optimizer and Sparse Categorical Crossentropy loss function. Training is performed for up to 30 epochs with early stopping to avoid overfitting.

---

### 5. 📊 Evaluate the Model

The model is evaluated on the test data to measure its accuracy. The predictions for test images are generated and visualized.

---

### 6. 🎨 Visualize Predictions

The script generates visualizations for:
- Displaying a predicted image and its corresponding class.
- Plotting a bar chart for prediction probabilities.
- Showing multiple images and their predictions in a grid layout.

---

## 📂 Output Files

The following output is generated during the execution:
- **Test Accuracy**: Printed to the console after evaluation.
- **Predictions Visualizations**: PNG files showing predicted images and corresponding probability bar charts.

---

## ⚠️ Notes for Reviewers

- Ensure TensorFlow is installed and configured correctly in your local environment.
- The dataset is automatically loaded using TensorFlow, so an internet connection is required for the initial download.
- If you encounter issues with library versions, try updating to the latest versions.

---

## 💬 Contact & Acknowledgments

- Project by: G. Kamesh
- Email: kamesh743243@gmail.com

Special thanks to TensorFlow and Keras for providing easy-to-use APIs for building deep learning models and the Fashion MNIST dataset.

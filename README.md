# mnist# **MNIST Handwritten Digit Classification**

## **Overview**
This project demonstrates a simple neural network model built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model is trained to recognize digits (0–9) with high accuracy using a feed-forward architecture.

---

## **Table of Contents**
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Setup](#setup)
- [Results](#results)
- [Conclusion](#conclusion)

---

## **Dataset**
The MNIST dataset is a classic benchmark in machine learning and consists of grayscale images of handwritten digits, each labeled with its corresponding digit.  
- **Number of Training Samples**: 60,000  
- **Number of Testing Samples**: 10,000  
- **Image Dimensions**: 28x28 pixels  
- **Number of Classes**: 10 (Digits: 0-9)

Source: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

## **Project Workflow**
1. **Data Preparation**:
   - Load the MNIST dataset using `tensorflow.keras.datasets`.
   - Normalize the pixel values to the range `[0, 1]` for better convergence.
   - Convert the labels to one-hot encoded vectors.

2. **Exploratory Data Analysis**:
   - Visualize the first few samples from the training dataset using `matplotlib`.

3. **Model Development**:
   - Define a simple feed-forward neural network with the following architecture:
     - **Flatten**: Converts the 28x28 input image to a 1D array.
     - **Dense Layer**: 128 neurons with ReLU activation.
     - **Output Layer**: 10 neurons with softmax activation.
   - Compile the model using:
     - Optimizer: **Adam**
     - Loss Function: **Categorical Crossentropy**
     - Metric: **Accuracy**

4. **Training**:
   - Train the model for 5 epochs with a batch size of 32.
   - Use 20% of the training data for validation.

5. **Evaluation**:
   - Evaluate the model's performance on the test set.
   - Display test loss and accuracy.

6. **Prediction**:
   - Visualize the predictions for the first 5 test samples alongside the ground truth.

---

## **Setup**

### Prerequisites
- Python 3.6 or higher
- TensorFlow 2.x
- Matplotlib
- NumPy

### Install Required Libraries
To install the dependencies, run:

```bash
pip install tensorflow matplotlib numpy
```

## **Running the Code**

### Clone the repository:
```bash
git clone <(https://github.com/mell0wO/mnist)>
cd <mnist>
```
### Run the script:
```bash
python mnist_classification.py
```
## Results

### Test Accuracy
Achieved approximately **98% accuracy** on the test set.

### Visualizations
- **Example Predictions**: Displayed handwritten digits alongside their predicted labels for better interpretability.
- **Loss and Accuracy Curves**: Illustrated the model's performance on training and validation data.

### Conclusion
This project successfully classifies handwritten digits from the MNIST dataset with high accuracy. The simple feed-forward architecture demonstrates how to implement and train a neural network for image classification tasks. 

**Potential Improvements**:
- Incorporate **convolutional layers (CNNs)** for enhanced performance.
- Experiment with deeper architectures to capture more complex features.

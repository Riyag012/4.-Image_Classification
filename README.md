## Facial Sentiment Analysis

This project focuses on binary classification to identify facial expressions as either happy or sad. It employs a Convolutional Neural Network (CNN) built with 
TensorFlow/Keras to achieve high accuracy in emotion detection. The system is designed for use in applications like human-computer interaction, sentiment tracking, and 
emotion-driven analytics.

## Dataset

- The dataset consists of labeled facial images categorized into two classes: happy and sad. 
- The model can be trained on any custom dataset with images organized into respective folders. 
- Ensure the dataset has balanced classes to avoid bias in predictions.
- You can use your own dataset by organizing images into two subfolders (`happy` and `sad`) within a root directory.
- Ensure images are resized and normalized to fit the model's input requirements (e.g., 256x256 resolution).


## Technologies Used

- Programming Language: Python 3.7+
  
- Libraries and Frameworks:
    - TensorFlow/Keras
    - NumPy
    - Matplotlib
    - OpenCV
    
- Tools:
    - Jupyter Notebook
    - TensorBoard for visualization
    - Git for version control


## Features
- Binary classification of facial expressions.
- Custom CNN for image-based sentiment detection.
- Accuracy evaluation and visualization through TensorBoard.
- Preprocessing pipeline for cleaning and normalizing image datasets


## Model Architecture
The CNN consists of:
- **3 Convolutional Layers**: Extract hierarchical features from images.
- **MaxPooling Layers**: Reduce spatial dimensions to prevent overfitting.
- **Fully Connected Dense Layer**: Aggregate extracted features.
- **Output Layer**: Single neuron with sigmoid activation for binary classification.
  

| Layer Type     | Filters | Kernel Size | Activation | Output Shape       |
|----------------|---------|-------------|------------|--------------------|
| Conv2D         | 16      | (3, 3)      | ReLU       | (256, 256, 16)     |
| MaxPooling2D   | -       | (2, 2)      | -          | (128, 128, 16)     |
| Conv2D         | 32      | (3, 3)      | ReLU       | (128, 128, 32)     |
| MaxPooling2D   | -       | (2, 2)      | -          | (64, 64, 32)       |
| Conv2D         | 16      | (3, 3)      | ReLU       | (64, 64, 16)       |
| MaxPooling2D   | -       | (2, 2)      | -          | (32, 32, 16)       |
| Flatten        | -       | -           | -          | (16384)            |
| Dense          | 256     | -           | ReLU       | (256)              |
| Dense          | 1       | -           | Sigmoid    | (1)                |


## Results

The CNN achieved:

- Training Accuracy: 100%
- Validation Accuracy: 81.49%
- Visualizations of the model's training performance (accuracy and loss) can be explored using TensorBoard logs.

## Future Enhancements
- Add real-time emotion detection using a webcam feed.
- Expand the model to classify additional emotions (e.g., angry, surprised).
- Improve accuracy with a larger and more diverse dataset.
- Experiment with transfer learning using pre-trained models like VGG16 or ResNet.

## Usage
- Train the model with your dataset by placing images into labeled folders (happy and sad) within a root directory.
- Evaluate its performance using provided test data or custom images.

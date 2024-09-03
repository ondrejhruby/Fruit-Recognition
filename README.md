# Fruit Recognition using CNN and Pre-trained Models

This project showcases a comprehensive deep learning approach to fruit recognition using Convolutional Neural Networks (CNN) and several pre-trained models, including ResNet, VGG16, VGG19, and Inception. The goal is to classify various types of fruits using a custom CNN model and compare it with state-of-the-art pre-trained architectures.

## Project Overview

The repository includes the following approaches:

- **Custom CNN Model**: A baseline model built from scratch using Convolutional Neural Networks to classify fruit images.
- **ResNet**: Implementation of the ResNet architecture, known for its deep residual connections, used for fine-tuning on the fruit dataset.
- **VGG16 and VGG19**: Two popular deep CNN architectures, fine-tuned for fruit classification, highlighting the effect of deeper networks.
- **Inception**: A pre-trained Inception model, demonstrating advanced feature extraction and classification capabilities.

## Dataset

The project uses a comprehensive fruit dataset with images of various fruit classes. The dataset is preprocessed and augmented to enhance model performance and generalization.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Fruit-Recognition.git
   cd Fruit-Recognition
   ```
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```
3. Explore the different models:

- Run the custom CNN: baseline_cnn.ipynb
- Run ResNet: resnet.ipynb
- Run VGG16: vgg16.ipynb
- Run VGG19: vgg19.ipynb
- Run Inception: inception.ipynb


## Key Results
- Detailed performance comparison between the custom CNN model and various pre-trained models.
- Insights into how transfer learning with pre-trained models outperforms traditional CNNs in terms of accuracy and efficiency.
- Visualizations of the training process, including accuracy and loss curves.

## Skills Learned
- Deep Learning with CNNs: Built and trained CNN models from scratch.
- Transfer Learning: Implemented and fine-tuned pre-trained models (ResNet, VGG, Inception) for a custom classification task.
- Data Preprocessing and Augmentation: Improved model performance through data augmentation techniques.
- Performance Evaluation: Analyzed and compared models using metrics such as accuracy, confusion matrices, and more.

## Conclusion
This project highlights the power of deep learning and transfer learning in image classification tasks. By comparing different models, it provides insights into how advanced architectures can significantly boost performance over baseline CNN models.

## Future Work
Experiment with other pre-trained models like EfficientNet or MobileNet.
Explore ensemble methods to combine the strengths of multiple models.
Implement real-time fruit recognition using the best-performing model.

## Acknowledgements

TensorFlow and Keras: For building and training deep learning models.

## Author
Ondrej Hruby

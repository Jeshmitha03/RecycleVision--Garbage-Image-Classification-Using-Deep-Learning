# RecycleVision – Garbage Image Classification Using Deep Learning

## 1. Project Overview

RecycleVision is a deep-learning-based computer vision project designed to automatically classify garbage images into different waste categories.

The project uses image preprocessing, data augmentation, Convolutional Neural Networks (CNN), transfer learning, model fine-tuning, evaluation metrics, and a Streamlit web application.

The final system uses **EfficientNetB0 with transfer learning** and achieves:

- **Test Accuracy: 96.22%**
- **Macro F1-Score: 94.88%**
- **Weighted F1-Score: 96.24%**

The application allows a user to upload a garbage image and receive:

- Predicted garbage category
- Prediction confidence
- Top-3 predicted categories

---

# 2. Project Objectives

The main objectives are:

1. Classify garbage images automatically.
2. Build a complete image-classification pipeline.
3. Handle class imbalance.
4. Compare a custom CNN with pretrained transfer-learning models.
5. Improve classification performance using fine-tuning.
6. Evaluate models using multiple classification metrics.
7. Identify common classification errors.
8. Deploy the final model through a Streamlit application.

---

# 3. Problem Statement

Manual garbage sorting is time-consuming and can be inconsistent.

Different types of waste can have similar visual characteristics. For example, different types of glass, plastic and metal objects can sometimes look similar.

RecycleVision attempts to automate this classification process using deep learning.

The system receives an image as input and predicts the most likely garbage category.

---

# 4. Domain

- Waste Management
- Environmental Technology
- Computer Vision
- Deep Learning
- Artificial Intelligence
- Image Classification

---

# 5. Technologies Used

## Programming Language

- Python

## Deep Learning

- TensorFlow
- Keras

## Machine Learning

- Scikit-learn

## Data Processing

- Pandas
- NumPy

## Image Processing

- Pillow
- TensorFlow image processing

## Visualization

- Matplotlib
- Seaborn

## Deployment

- Streamlit

## Development Environment

- Jupyter Notebook
- VS Code
- Python Virtual Environment

---

# 6. Dataset

The project uses a garbage image classification dataset containing **15,515 valid images**.

The dataset contains 12 garbage categories.

## Classes

| Index | Class |
|---:|---|
| 0 | battery |
| 1 | biological |
| 2 | brown-glass |
| 3 | cardboard |
| 4 | clothes |
| 5 | green-glass |
| 6 | metal |
| 7 | paper |
| 8 | plastic |
| 9 | shoes |
| 10 | trash |
| 11 | white-glass |

## Class Distribution

| Class | Images |
|---|---:|
| battery | 945 |
| biological | 985 |
| brown-glass | 607 |
| cardboard | 891 |
| clothes | 5325 |
| green-glass | 629 |
| metal | 769 |
| paper | 1050 |
| plastic | 865 |
| shoes | 1977 |
| trash | 697 |
| white-glass | 775 |

The dataset is imbalanced because some categories contain significantly more images than others.

For example:

- Clothes: 5,325 images
- Brown-glass: 607 images

Therefore, class imbalance needed to be considered during training.

---

# 7. Exploratory Data Analysis

Before training the models, the dataset was inspected to understand its structure and quality.

The following were analyzed:

- Number of images
- Class distribution
- Image dimensions
- Image color modes
- Corrupted images
- Very small images

## Dataset Size

Total images:

```text
15,515
```

## Image Dimensions

Images were found in multiple dimensions.

Some common dimensions were:

- 400 × 533
- 512 × 384
- 225 × 225
- 400 × 534
- 275 × 183

Because the images had different dimensions, resizing was required before model training.

## Color Modes

Most images were RGB.

Approximately:

- RGB: 15,481
- P: 34

Images were converted to 3-channel RGB during preprocessing.

## Corrupted Images

No corrupted images were detected.

```text
Corrupted images = 0
```

## Very Small Images

No images below the selected minimum size threshold were found.

---

# 8. Data Splitting

The dataset was divided into three subsets.

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 10,860 | 70% |
| Validation | 2,327 | 15% |
| Testing | 2,328 | 15% |
| Total | 15,515 | 100% |

A **stratified split** was used.

This helps preserve the class distribution across training, validation and test datasets.

The test dataset was kept separate and was not used during model training.

---

# 9. Image Preprocessing

Images were converted into a consistent format before being provided to the deep-learning models.

The main preprocessing steps were:

1. Read image
2. Decode image
3. Convert to RGB
4. Resize image
5. Convert pixel values to floating-point representation
6. Apply model-specific preprocessing when required

The target image size was:

```text
224 × 224 × 3
```

The `3` represents the RGB color channels.

---

# 10. Why Resize Images?

The original images have different dimensions.

Deep-learning models require a consistent input shape.

Therefore, all images were resized to:

```text
224 × 224
```

This also matches the input size used by the pretrained transfer-learning models.

---

# 11. Pixel Normalization / Model Preprocessing

For the baseline CNN, pixel values were converted from:

```text
0 – 255
```

to:

```text
0 – 1
```

using:

```python
image = image / 255.0
```

For the EfficientNetB0 pipeline, model-specific preprocessing behavior was used rather than manually applying the baseline `/255.0` transformation.

This keeps the input processing consistent with the pretrained model.

---

# 12. Data Augmentation

Data augmentation was applied to the training images.

The augmentation pipeline included:

```python
RandomFlip("horizontal")
RandomRotation(0.10)
RandomZoom(0.10)
RandomContrast(0.10)
```

## Why Data Augmentation?

Data augmentation creates slightly modified versions of training images.

This helps the model learn more general visual patterns and reduces overfitting.

For example:

```text
Original image
      ↓
Horizontal flip
      ↓
Small rotation
      ↓
Small zoom
      ↓
Contrast variation
```

Augmentation was applied to training data and not to validation/test data.

---

# 13. Handling Class Imbalance

The dataset contains unequal numbers of images in different classes.

To address this, **class weights** were calculated using:

```python
compute_class_weight(
    class_weight="balanced"
)
```

Minority classes received higher weights, while majority classes received lower weights.

This encourages the model to pay more attention to underrepresented classes.

For example:

```text
Brown-glass → higher weight
Clothes     → lower weight
```

Class weighting was used during model training.

---

# 14. TensorFlow Data Pipeline

TensorFlow `tf.data.Dataset` was used for efficient data loading.

The pipeline was:

```text
Image Path
    ↓
Read Image
    ↓
Decode RGB
    ↓
Resize
    ↓
Preprocessing
    ↓
Shuffle Training Data
    ↓
Batch Size = 32
    ↓
Prefetch
    ↓
Model
```

The training dataset was shuffled using a fixed random seed.

Prefetching was used to improve the data-loading pipeline.

---

# 15. Baseline CNN

A custom CNN model was developed as the baseline.

Architecture:

```text
Input 224 × 224 × 3
        ↓
Data Augmentation
        ↓
Conv2D 32
        ↓
MaxPooling
        ↓
Conv2D 64
        ↓
MaxPooling
        ↓
Conv2D 128
        ↓
MaxPooling
        ↓
Global Average Pooling
        ↓
Dense 128
        ↓
Dense 12 Softmax
```

The model was trained using:

- Adam optimizer
- Learning rate = 0.001
- Sparse categorical cross-entropy
- Class weights
- Early stopping
- Learning-rate reduction

## Baseline Result

Test accuracy:

```text
67.44%
```

Macro F1-score:

```text
63.28%
```

This established the baseline for comparison with transfer-learning models.

---

# 16. Transfer Learning

After establishing the baseline, pretrained CNN architectures were used.

Transfer learning means using a model that has already learned useful visual features from a large dataset such as ImageNet.

Instead of training an entire deep network from scratch, the pretrained model is used as a feature extractor and a new classification head is added for the garbage classes.

Advantages include:

- Faster training
- Better feature extraction
- Better performance with limited data
- Reduced need to train millions of parameters from scratch

---

# 17. MobileNetV2

MobileNetV2 pretrained on ImageNet was used.

Initially, the pretrained base model was frozen.

A custom classification head was added:

```text
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Dense 128
     ↓
Dense 12 Softmax
```

## MobileNetV2 Result

Test accuracy:

```text
92.83%
```

Macro F1-score:

```text
89.85%
```

This was a major improvement over the baseline CNN.

---

# 18. MobileNetV2 Fine-Tuning

After feature extraction, the MobileNetV2 model was fine-tuned.

The last part of the pretrained network was made trainable while most earlier layers remained frozen.

Batch Normalization layers were kept frozen to improve training stability.

A smaller learning rate was used:

```text
0.00001
```

## Fine-Tuned Result

Test accuracy:

```text
93.47%
```

Macro F1-score:

```text
90.60%
```

Fine-tuning improved the model compared with the original MobileNetV2.

---

# 19. EfficientNetB0

EfficientNetB0 was then evaluated using transfer learning.

The pretrained ImageNet model was initially frozen.

Architecture:

```text
Input
  ↓
Data Augmentation
  ↓
EfficientNetB0
  ↓
Global Average Pooling
  ↓
Dropout 0.3
  ↓
Dense 128
  ↓
Dense 12 Softmax
```

Training configuration:

- Optimizer: Adam
- Learning rate: 0.0001
- Loss: Sparse categorical cross-entropy
- Batch size: 32
- Epochs: 15
- Class weights: Yes
- Early stopping: Yes
- Reduce learning rate on plateau: Yes

---

# 20. EfficientNetB0 Results

The final EfficientNetB0 model achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **96.22%** |
| Macro Precision | 95.01% |
| Macro Recall | 94.81% |
| Macro F1 | **94.88%** |
| Weighted F1 | **96.24%** |

The model was evaluated on 2,328 unseen test images.

---

# 21. Class-Wise Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| battery | 0.9710 | 0.9437 | 0.9571 |
| biological | 0.9862 | 0.9662 | 0.9761 |
| brown-glass | 0.9556 | 0.9451 | 0.9503 |
| cardboard | 0.9848 | 0.9701 | 0.9774 |
| clothes | 0.9924 | 0.9837 | 0.9881 |
| green-glass | 0.9681 | 0.9681 | 0.9681 |
| metal | 0.8548 | 0.9217 | 0.8870 |
| paper | 0.9490 | 0.9430 | 0.9460 |
| plastic | 0.8955 | 0.9231 | 0.9091 |
| shoes | 0.9516 | 0.9933 | 0.9720 |
| trash | 0.9794 | 0.9135 | 0.9453 |
| white-glass | 0.9130 | 0.9052 | 0.9091 |

The strongest class performance was obtained for:

- Clothes
- Cardboard
- Biological
- Shoes

Some comparatively difficult classes were:

- Metal
- Plastic
- White-glass

---

# 22. Model Comparison

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Baseline CNN | 67.44% | 63.28% |
| MobileNetV2 | 92.83% | 89.85% |
| MobileNetV2 Fine-tuned | 93.47% | 90.60% |
| **EfficientNetB0** | **96.22%** | **94.88%** |

EfficientNetB0 was selected as the final model because it achieved the best test accuracy and macro F1-score.

Improvement over the baseline CNN:

```text
96.22% - 67.44%
= 28.78 percentage points
```

Improvement over fine-tuned MobileNetV2:

```text
96.22% - 93.47%
= 2.75 percentage points
```

---

# 23. Confusion Matrix

A confusion matrix was generated for EfficientNetB0.

The confusion matrix helps identify:

- Correct predictions
- Incorrect predictions
- Frequently confused classes

Rows represent actual classes and columns represent predicted classes.

This is particularly useful for identifying visually similar garbage categories.

---

# 24. Error Analysis

Misclassified test images were extracted and analyzed.

The analysis included:

- True class
- Predicted class
- Image path
- Number of errors by class
- Common confusion pairs

The purpose of error analysis is to understand where the model makes mistakes rather than relying only on overall accuracy.

---

# 25. Model Saving

The final EfficientNetB0 model was saved as:

```text
models/efficientnetb0.keras
```

Model information was saved as:

```text
models/model_info.json
```

The model information contains:

- Model name
- Number of classes
- Image size
- Test accuracy
- Macro F1
- Weighted F1
- Class mapping

---

# 26. Streamlit Application

A Streamlit application was developed to deploy the trained model.

The application is located at:

```text
dashboard/app.py
```

The user can:

1. Upload a garbage image.
2. View the uploaded image.
3. Get the predicted garbage class.
4. View the prediction confidence.
5. View the Top-3 predictions.
6. View the supported garbage categories.

---

# 27. Streamlit Prediction Pipeline

The dashboard follows this workflow:

```text
Upload Image
      ↓
Convert to RGB
      ↓
Resize to 224 × 224
      ↓
EfficientNetB0
      ↓
Prediction Probabilities
      ↓
Sort Probabilities
      ↓
Top-3 Predictions
      ↓
Display Class + Confidence
```

---

# 28. Running the Application

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

From the project root, run:

```powershell
streamlit run dashboard\app.py
```

The application opens in the browser.

---

# 29. Project Directory Structure

```text
GUVI_my_Fourth_project/
│
├── data/
│   ├── garbage_classification/
│   ├── processed/
│   └── class_mapping.json
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Baseline_CNN.ipynb
│   ├── 04_MobileNetV2.ipynb
│   └── 05_EfficientNetB0.ipynb
│
├── models/
│   ├── baseline_cnn.keras
│   ├── mobilenetv2.keras
│   ├── mobilenetv2_finetuned.keras
│   ├── efficientnetb0.keras
│   └── model_info.json
│
├── evaluation/
│   ├── model_comparison.csv
│   ├── model_accuracy_comparison.png
│   ├── efficientnetb0_classification_report.txt
│   ├── efficientnetb0_confusion_matrix.png
│   ├── efficientnetb0_training_accuracy.png
│   ├── efficientnetb0_training_loss.png
│   ├── misclassified_images.csv
│   └── error_analysis_summary.csv
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
│
├── documentation.md
│
└── .venv/
```

The `.venv` directory should not be uploaded to GitHub.

---

# 30. Key Results

The final project achieved:

```text
Total Images       : 15,515
Number of Classes  : 12
Image Size         : 224 × 224
Training Images    : 10,860
Validation Images  : 2,327
Test Images        : 2,328

Final Model        : EfficientNetB0

Test Accuracy      : 96.22%
Macro F1-Score     : 94.88%
Weighted F1-Score  : 96.24%
```

---

# 31. Business Use Cases

RecycleVision can be extended to several real-world applications.

## Smart Waste Bins

Cameras can classify waste before it enters different compartments.

## Automated Waste Sorting

The model can be integrated with robotic or conveyor-based sorting systems.

## Municipal Waste Management

Waste-management organizations can use image-based classification to support automated sorting.

## Educational Applications

The system can teach users how to identify different types of recyclable and non-recyclable waste.

## Environmental Analytics

Predicted waste categories can be collected over time to understand waste-generation patterns.

---

# 32. Advantages

- High classification accuracy
- Uses transfer learning
- Handles class imbalance
- Uses image augmentation
- Supports 12 garbage categories
- Provides Top-3 predictions
- Interactive Streamlit interface
- Modular project structure
- Multiple models were compared

---

# 33. Limitations

Although the model achieves high test accuracy, some limitations remain.

1. Performance depends on the quality of the input image.
2. Images significantly different from the training dataset may produce incorrect predictions.
3. Visually similar categories can still be confused.
4. The dataset may not represent every real-world waste environment.
5. The current application performs image classification rather than object detection.
6. The model does not identify multiple garbage objects separately within one image.

---

# 34. Future Improvements

Future versions could include:

- Object detection for multiple waste items
- Real-time camera classification
- Mobile application deployment
- TensorFlow Lite deployment
- Additional garbage categories
- Larger and more diverse datasets
- Explainable AI techniques
- Edge-device deployment
- Automated waste sorting hardware integration

---

# 35. Conclusion

RecycleVision demonstrates a complete deep-learning workflow for garbage image classification.

The project progressed from exploratory data analysis and preprocessing to a baseline CNN, followed by transfer learning using MobileNetV2 and EfficientNetB0.

The final EfficientNetB0 model achieved:

**96.22% test accuracy and 94.88% macro F1-score.**

The model was integrated into a Streamlit application that allows users to upload garbage images and receive predictions with confidence scores and Top-3 results.

The project demonstrates practical skills in:

- Python
- TensorFlow/Keras
- CNN
- Computer Vision
- Image Preprocessing
- Data Augmentation
- Transfer Learning
- Fine-Tuning
- Model Evaluation
- Streamlit Deployment

Overall, RecycleVision provides a strong foundation for an automated waste-classification system and demonstrates how deep learning can be applied to environmental and waste-management problems.
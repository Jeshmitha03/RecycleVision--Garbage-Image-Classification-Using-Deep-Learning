# RecycleVision--Garbage-Image-Classification-Using-Deep-Learning
RecycleVision – Garbage Image Classification Using Deep Learning is a deep learning project that uses Convolutional Neural Networks (CNNs) and transfer learning to automatically classify garbage images into 12 waste categories, helping improve waste segregation and support efficient recycling and waste management.
# ♻️ RecycleVision – Garbage Image Classification Using Deep Learning

## 📌 Project Overview

**RecycleVision** is a deep-learning-based computer vision project designed to automatically classify garbage images into different waste categories.

The project uses **Convolutional Neural Networks (CNNs)** and **Transfer Learning** techniques to build and compare multiple image-classification models.

The best-performing model, **EfficientNetB0**, achieved:

- 🎯 **96.22% Test Accuracy**
- 📊 **94.88% Macro F1 Score**
- 📈 **96.24% Weighted F1 Score**

A **Streamlit web application** was also developed to allow users to upload a garbage image and receive the predicted waste category along with the prediction confidence.

---

## 🎯 Objectives

The main objectives of this project are:

1. Classify garbage images into 12 different waste categories.
2. Perform image data preprocessing and augmentation.
3. Handle class imbalance using class weights.
4. Build a baseline CNN model.
5. Apply transfer learning using MobileNetV2.
6. Fine-tune MobileNetV2 for improved performance.
7. Apply transfer learning using EfficientNetB0.
8. Compare the performance of different models.
9. Analyze model errors using confusion matrices and misclassified images.
10. Deploy the final model using Streamlit.

---

## 🗑️ Waste Categories

The dataset contains **12 garbage categories**:

| Class | Waste Category |
|---:|---|
| 0 | Battery |
| 1 | Biological |
| 2 | Brown Glass |
| 3 | Cardboard |
| 4 | Clothes |
| 5 | Green Glass |
| 6 | Metal |
| 7 | Paper |
| 8 | Plastic |
| 9 | Shoes |
| 10 | Trash |
| 11 | White Glass |

---

## 📊 Dataset

The dataset contains:

- **15,515 images**
- **12 classes**
- RGB and palette-format images
- Different original image dimensions
- No corrupted images detected
- Images resized to **224 × 224 pixels**

### Dataset Split

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 10,860 | 70% |
| Validation | 2,327 | 15% |
| Testing | 2,328 | 15% |
| **Total** | **15,515** | **100%** |

The dataset was split using **stratified sampling** so that the class distribution was maintained across training, validation, and testing datasets.

---

## 🔎 Exploratory Data Analysis

The dataset was analyzed before model training.

The following were checked:

- Number of images
- Number of classes
- Class distribution
- Image dimensions
- Image color modes
- Corrupted images
- Very small images
- Class imbalance

### Dataset Quality

The analysis found:

- **15,515 valid images**
- **0 corrupted images**
- **0 images below 50 pixels**
- Most images were RGB
- Image dimensions varied significantly

Because the images had different dimensions, they were resized to a common size of **224 × 224**.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied.

### 1. Image Resizing

All images were resized to:

```text
224 × 224 × 3
```

This provides a consistent input shape for the deep-learning models.

### 2. RGB Conversion

Images were converted to RGB format so that all images had three color channels:

```text
Red
Green
Blue
```

### 3. Data Augmentation

Data augmentation was used to increase the variety of training images.

The following transformations were used:

- Random horizontal flipping
- Random rotation
- Random zoom
- Random contrast

Example:

```python
tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10)
])
```

### 4. Class Imbalance Handling

The dataset was imbalanced. For example, the **clothes** class contained significantly more images than some glass categories.

To reduce the effect of class imbalance, **class weights** were calculated using:

```python
compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label"].values
)
```

The calculated weights were provided during model training.

---

# 🧠 Models Used

Three main approaches were evaluated.

## 1. Baseline CNN

A custom CNN was created as the baseline model.

Architecture:

```text
Input Image
    ↓
Data Augmentation
    ↓
Conv2D – 32 filters
    ↓
MaxPooling
    ↓
Conv2D – 64 filters
    ↓
MaxPooling
    ↓
Conv2D – 128 filters
    ↓
MaxPooling
    ↓
Global Average Pooling
    ↓
Dense – 128
    ↓
Softmax – 12 Classes
```

### Result

**Test Accuracy: 67.44%**

The baseline model provided a reference point for evaluating transfer-learning models.

---

# 📱 2. MobileNetV2

MobileNetV2 pretrained on ImageNet was used for transfer learning.

The pretrained convolutional base was initially frozen, and a custom classification head was added.

### Feature Extraction Result

**Test Accuracy: 92.83%**

**Macro F1: 89.85%**

**Weighted F1: 92.92%**

---

## 🔧 MobileNetV2 Fine-Tuning

After feature extraction, the final layers of MobileNetV2 were unfrozen and trained with a smaller learning rate.

The learning rate was reduced to:

```text
0.00001
```

Batch Normalization layers were kept frozen during fine-tuning.

### Result

**Test Accuracy: 93.47%**

**Macro F1: 90.60%**

**Weighted F1: 93.54%**

Fine-tuning improved the model compared with the original MobileNetV2 model.

---

# 🚀 3. EfficientNetB0

EfficientNetB0 was used as the final transfer-learning model.

The model was pretrained on ImageNet and adapted for the 12 garbage classes.

Architecture:

```text
Input Image
    ↓
Data Augmentation
    ↓
EfficientNetB0
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense Layer – 128
    ↓
Softmax – 12 Classes
```

Class weights were also applied during training.

### Final Result

**Test Accuracy: 96.22%**

**Macro F1 Score: 94.88%**

**Weighted F1 Score: 96.24%**

EfficientNetB0 was selected as the final model because it achieved the best overall performance.

---

# 📈 Model Comparison

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Baseline CNN | 67.44% | 63.28% | 66.53% |
| MobileNetV2 | 92.83% | 89.85% | 92.92% |
| MobileNetV2 Fine-tuned | 93.47% | 90.60% | 93.54% |
| **EfficientNetB0** | **96.22%** | **94.88%** | **96.24%** |

### Performance Improvement

EfficientNetB0 improved test accuracy by approximately:

- **28.78 percentage points** over the baseline CNN
- **2.75 percentage points** over fine-tuned MobileNetV2

This demonstrates the advantage of using pretrained deep-learning architectures for image classification.

---

# 📊 EfficientNetB0 Class-wise Performance

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Battery | 0.9710 | 0.9437 | 0.9571 |
| Biological | 0.9862 | 0.9662 | 0.9761 |
| Brown Glass | 0.9556 | 0.9451 | 0.9503 |
| Cardboard | 0.9848 | 0.9701 | 0.9774 |
| Clothes | 0.9924 | 0.9837 | 0.9881 |
| Green Glass | 0.9681 | 0.9681 | 0.9681 |
| Metal | 0.8548 | 0.9217 | 0.8870 |
| Paper | 0.9490 | 0.9430 | 0.9460 |
| Plastic | 0.8955 | 0.9231 | 0.9091 |
| Shoes | 0.9516 | 0.9933 | 0.9720 |
| Trash | 0.9794 | 0.9135 | 0.9453 |
| White Glass | 0.9130 | 0.9052 | 0.9091 |

---

# 🔍 Error Analysis

After testing the final model, misclassified images were analyzed.

The analysis included:

- Confusion matrix
- Misclassified images
- True class vs predicted class
- Confusion pairs
- Class-wise error analysis

This helped identify classes that were more difficult for the model.

The relatively lower performance of some classes, such as **metal, plastic, and white glass**, indicates that some waste categories have visually similar characteristics.

---

# 💾 Model Files

The trained models were saved in the `models` directory.

```text
models/
├── baseline_cnn.keras
├── mobilenetv2.keras
├── mobilenetv2_finetuned.keras
├── efficientnetb0.keras
└── model_info.json
```

The final deployment model is:

```text
efficientnetb0.keras
```

---

# 🌐 Streamlit Application

A Streamlit web application was created for real-time garbage image classification.

The application allows the user to:

1. Upload an image.
2. Display the uploaded image.
3. Run the EfficientNetB0 model.
4. Display the predicted waste category.
5. Display prediction confidence.
6. Display the top-3 predictions.

### Application Flow

```text
User Uploads Image
        ↓
Convert to RGB
        ↓
Resize to 224 × 224
        ↓
Prepare Tensor
        ↓
EfficientNetB0
        ↓
Prediction Probabilities
        ↓
Top-3 Predictions
        ↓
Display Result
```

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd GUVI_my_Fourth_project
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Streamlit

From the project root:

```powershell
streamlit run dashboard\app.py
```

The Streamlit application will open in your browser.

---

# 📁 Project Structure

```text
GUVI_my_Fourth_project/
│
├── data/
│   ├── garbage_classification/
│   │   ├── battery/
│   │   ├── biological/
│   │   ├── brown-glass/
│   │   ├── cardboard/
│   │   ├── clothes/
│   │   ├── green-glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   ├── shoes/
│   │   ├── trash/
│   │   └── white-glass/
│   │
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   │
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
│   ├── efficientnetb0_classification_report.txt
│   ├── efficientnetb0_training_accuracy.png
│   ├── efficientnetb0_training_loss.png
│   ├── efficientnetb0_confusion_matrix.png
│   ├── model_comparison.csv
│   ├── model_accuracy_comparison.png
│   ├── misclassified_images.csv
│   └── error_analysis_summary.csv
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── documentation.md
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- CNN
- Transfer Learning
- EfficientNetB0
- MobileNetV2

### Data Processing

- NumPy
- Pandas
- Pillow

### Machine Learning

- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Deployment

- Streamlit

### Development

- Jupyter Notebook
- VS Code

---

# 🌱 Potential Applications

RecycleVision can be used as a foundation for:

- Automated waste sorting
- Smart recycling systems
- Waste management systems
- Recycling-center automation
- Educational applications
- Smart-bin systems
- Environmental monitoring
- Computer-vision-based recycling solutions

---

# ⚠️ Limitations

Although the model achieved high test accuracy, there are some limitations:

1. Performance depends on image quality.
2. Images outside the 12 trained categories may be incorrectly classified.
3. Real-world waste can contain multiple objects in a single image.
4. Some waste categories have visually similar characteristics.
5. Dataset images may not fully represent real-world lighting and backgrounds.
6. The model performs image classification rather than object detection or segmentation.

---

# 🚀 Future Improvements

Future versions of the project could include:

- Object detection for multiple waste items.
- Waste segmentation.
- Real-time camera-based classification.
- Larger and more diverse datasets.
- Additional waste categories.
- Mobile application deployment.
- Cloud deployment.
- Model quantization for edge devices.
- Explainable AI techniques.
- Real-time smart-bin integration.

---

# 🏆 Key Results

The final RecycleVision system achieved:

```text
Dataset              : 15,515 images
Number of Classes    : 12
Image Size           : 224 × 224
Final Model          : EfficientNetB0
Test Accuracy        : 96.22%
Macro F1 Score       : 94.88%
Weighted F1 Score     : 96.24%
Deployment           : Streamlit
```

---

# 💡 Conclusion

RecycleVision demonstrates a complete deep-learning workflow for garbage image classification.

The project started with a custom CNN baseline and progressively improved performance using transfer learning and fine-tuning.

The final **EfficientNetB0 model achieved 96.22% test accuracy and a 94.88% macro F1 score**, making it the best-performing model among the evaluated approaches.

The project also includes error analysis, model comparison, model saving, and a Streamlit deployment interface, providing an end-to-end computer vision solution for waste classification.

---

## 👩‍💻 Author

**Jeshmitha J**

GitHub: `Jeshmitha03`

---

## ⭐ Project Highlights

- ✅ 15,515 real garbage images
- ✅ 12 waste categories
- ✅ Complete EDA
- ✅ Stratified train/validation/test split
- ✅ Image preprocessing
- ✅ Data augmentation
- ✅ Class imbalance handling
- ✅ Custom CNN baseline
- ✅ MobileNetV2 transfer learning
- ✅ MobileNetV2 fine-tuning
- ✅ EfficientNetB0 transfer learning
- ✅ 96.22% test accuracy
- ✅ Error analysis
- ✅ Model comparison
- ✅ Streamlit deployment
- ✅ Complete project documentation

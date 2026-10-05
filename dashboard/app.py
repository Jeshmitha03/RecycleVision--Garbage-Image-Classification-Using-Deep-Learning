import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "efficientnetb0.keras"
)

MODEL_INFO_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "model_info.json"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RecycleVision",
    page_icon="♻️",
    layout="wide"
)


# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

with open(MODEL_INFO_PATH, "r") as f:
    model_info = json.load(f)


index_to_class = {
    int(k): v
    for k, v in model_info["classes"].items()
}

class_names = [
    index_to_class[i]
    for i in range(
        len(index_to_class)
    )
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image = np.array(
        image
    ).astype(
        np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    processed_image = preprocess_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    results = []

    for index in top_indices:

        results.append({
            "class": class_names[index],
            "confidence": float(
                predictions[index]
            )
        })

    return results


# ============================================================
# HEADER
# ============================================================

st.title(
    "♻️ RecycleVision"
)

st.subheader(
    "Garbage Image Classification Using Deep Learning"
)

st.write(
    "Upload an image of waste and the trained "
    "EfficientNetB0 model will predict its category."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "About the Model"
)

st.sidebar.write(
    "Model: EfficientNetB0"
)

st.sidebar.write(
    "Test Accuracy: 96.22%"
)

st.sidebar.write(
    "Macro F1-Score: 94.88%"
)

st.sidebar.write(
    "Image Size: 224 × 224"
)

st.sidebar.write(
    "Number of Classes: 12"
)


st.sidebar.subheader(
    "Supported Categories"
)

for class_name in class_names:

    st.sidebar.write(
        f"• {class_name}"
    )


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a garbage image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    col1, col2 = st.columns(
        2
    )

    # --------------------------------------------------------
    # Uploaded Image
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "Prediction"
        )

        results = predict_image(
            image
        )

        top_prediction = results[0]

        st.success(
            f"Predicted Class: "
            f"{top_prediction['class']}"
        )

        st.metric(
            "Confidence",
            f"{top_prediction['confidence'] * 100:.2f}%"
        )


        # ----------------------------------------------------
        # Top 3 Predictions
        # ----------------------------------------------------

        st.subheader(
            "Top-3 Predictions"
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            st.write(
                f"**{rank}. "
                f"{result['class']}** — "
                f"{result['confidence'] * 100:.2f}%"
            )

            st.progress(
                result["confidence"]
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "RecycleVision | Deep Learning Garbage Classification"
)
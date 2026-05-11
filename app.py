import streamlit as st
import numpy as np
import joblib
import cv2
import math
import os
import gdown
from PIL import Image, ImageDraw
import random
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from skimage.feature import hog
from streamlit_drawable_canvas import st_canvas

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Math Operator Classifier",
    page_icon="🔢",
    layout="centered"
)

# ── Title ──────────────────────────────────────────────────
st.title("🔢 Math Operator Symbol Classifier")
st.markdown("**Team:** Adithyan | Jia | Theertha")
st.markdown("Draw a math operator symbol or upload an image to classify it!")
st.markdown("---")

# ── Class mapping ──────────────────────────────────────────
CLASS_NAMES = {
    0: '+',
    1: '-',
    2: 'mul (×)',
    3: 'div (÷)',
    4: '=',
    5: '!=  (≠)',
    6: '<',
    7: '>',
    8: '+- (±)',
    9: 'sqrt (√)'
}

FULL_NAMES = {
    0: 'Addition (+)',
    1: 'Subtraction (−)',
    2: 'Multiplication (×)',
    3: 'Division (÷)',
    4: 'Equals (=)',
    5: 'Not Equal (≠)',
    6: 'Less Than (<)',
    7: 'Greater Than (>)',
    8: 'Plus Minus (±)',
    9: 'Square Root (√)'
}

# ── Load models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    os.makedirs('models', exist_ok=True)

    # Google Drive file IDs
    model_files = {
        'models/svm_model.pkl' : '1kElx3d0EHSGLjK6Hjzpx1PXcpavRqbgo',
        'models/knn_model.pkl' : '1W2iYcDO1d8fOODMzWkqIjqq_ipFDdpkN',
        'models/dt_model.pkl'  : '1Y6dl5igxCYOtEnG_GdGB81Ochywztttn',
        'models/scaler.pkl'    : '1idYVwLytco8QoDVXouqPM1KcidA-p93p',
    }

    for path, file_id in model_files.items():
        if not os.path.exists(path):
            url = 'https://drive.google.com/uc?id=' + file_id
            gdown.download(url, path, quiet=False)

    try:
        svm    = joblib.load('models/svm_model.pkl')
        knn    = joblib.load('models/knn_model.pkl')
        dt     = joblib.load('models/dt_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return svm, knn, dt, scaler, True
    except Exception as e:
        st.error('Error loading models: ' + str(e))
        return None, None, None, None, False

svm_model, knn_model, dt_model, scaler, models_loaded = load_models()

if not models_loaded:
    st.error("Models not found! Make sure models/ folder contains svm_model.pkl, knn_model.pkl, dt_model.pkl, scaler.pkl")
    st.stop()

st.success("Models loaded successfully!")

# ── Preprocessing pipeline ─────────────────────────────────
def preprocess_image(img_array):
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Otsu binarization
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Skeletonize
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    skeleton_img = img_as_ubyte(skeleton)

    # Resize to 32x32
    resized = cv2.resize(skeleton_img, (32, 32), interpolation=cv2.INTER_AREA)
    return resized

def extract_features(img_32x32):
    flat = img_32x32.flatten()

    # HOG features
    hog_feat = hog(
        img_32x32,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    # Projection profiles
    horizontal = img_32x32.sum(axis=1)
    vertical   = img_32x32.sum(axis=0)
    proj_feat  = np.concatenate([horizontal, vertical])

    # Run-length features
    runs = []
    for row in img_32x32:
        run_len = 0
        in_run  = False
        for pixel in row:
            if pixel > 0:
                run_len += 1
                in_run = True
            else:
                if in_run:
                    runs.append(run_len)
                    run_len = 0
                    in_run  = False
        if in_run:
            runs.append(run_len)

    max_runs = 32
    if len(runs) >= max_runs:
        runs = runs[:max_runs]
    else:
        runs = runs + [0] * (max_runs - len(runs))
    run_feat = np.array(runs)

    # Combine all features
    combined = np.concatenate([hog_feat, proj_feat, run_feat])
    return combined

def predict(img_array, model_choice):
    # Preprocess
    processed = preprocess_image(img_array)

    # Extract features
    features = extract_features(processed)

    # Scale
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    if model_choice == 'SVM (Best)':
        model = svm_model
    elif model_choice == 'KNN':
        model = knn_model
    else:
        model = dt_model

    pred_class = model.predict(features_scaled)[0]
    pred_proba = model.predict_proba(features_scaled)[0]

    return pred_class, pred_proba, processed

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ['SVM (Best)', 'KNN', 'Decision Tree'],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.markdown("| Model | Accuracy |")
st.sidebar.markdown("|-------|----------|")
st.sidebar.markdown("| SVM   | 99.29%   |")
st.sidebar.markdown("| KNN   | 98.86%   |")
st.sidebar.markdown("| DTree | 92.71%   |")

st.sidebar.markdown("---")
st.sidebar.markdown("### Symbols Supported")
for k, v in FULL_NAMES.items():
    st.sidebar.markdown("- " + v)

# ── Input method tabs ──────────────────────────────────────
tab1, tab2 = st.tabs(["✏️ Draw Symbol", "📁 Upload Image"])

# ── TAB 1: Drawing canvas ──────────────────────────────────
with tab1:
    st.markdown("### Draw your symbol below:")
    st.markdown("Use your mouse to draw. Keep the symbol centered and large.")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=12,
        stroke_color="black",
        background_color="white",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    col1, col2 = st.columns(2)
    with col1:
        classify_btn = st.button("🔍 Classify Drawing", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if classify_btn and canvas_result.image_data is not None:
        img_array = canvas_result.image_data.astype(np.uint8)

        # Check if canvas has anything drawn
        if img_array[:, :, :3].sum() < 255*300*300*3 - 1000:
            pred_class, pred_proba, processed = predict(img_array[:, :, :3], model_choice)

            st.markdown("---")
            st.markdown("## Result")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### Predicted Symbol")
                st.markdown(
                    "<h1 style='text-align:center; font-size:80px'>" +
                    CLASS_NAMES[pred_class] + "</h1>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown("### Full Name")
                st.info(FULL_NAMES[pred_class])
                st.markdown("### Confidence")
                confidence = pred_proba[pred_class]
                st.progress(float(confidence))
                st.markdown(str(round(confidence * 100, 2)) + "%")
            with col3:
                st.markdown("### Preprocessed")
                st.image(processed, width=100, caption="32x32 skeleton")

            st.markdown("---")
            st.markdown("### Class Probabilities")
            import pandas as pd
            prob_df = pd.DataFrame({
                'Symbol': list(FULL_NAMES.values()),
                'Probability': pred_proba
            }).sort_values('Probability', ascending=False)
            st.bar_chart(prob_df.set_index('Symbol')['Probability'])
        else:
            st.warning("Please draw a symbol first!")

# ── TAB 2: Image upload ────────────────────────────────────
with tab2:
    st.markdown("### Upload an image of a handwritten math symbol:")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a single math operator symbol"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            if st.button("🔍 Classify Image", use_container_width=True):
                pred_class, pred_proba, processed = predict(img_array, model_choice)

                st.markdown("### Result")
                st.markdown(
                    "<h1 style='text-align:center; font-size:60px'>" +
                    CLASS_NAMES[pred_class] + "</h1>",
                    unsafe_allow_html=True
                )
                st.success(FULL_NAMES[pred_class])
                confidence = pred_proba[pred_class]
                st.progress(float(confidence))
                st.markdown("Confidence: " + str(round(confidence * 100, 2)) + "%")
                st.image(processed, width=100, caption="Preprocessed 32x32")

        if uploaded_file and st.button("Show All Probabilities"):
            pred_class, pred_proba, processed = predict(img_array, model_choice)
            import pandas as pd
            prob_df = pd.DataFrame({
                'Symbol': list(FULL_NAMES.values()),
                'Probability': pred_proba
            }).sort_values('Probability', ascending=False)
            st.bar_chart(prob_df.set_index('Symbol')['Probability'])

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px'>"
    "Math Operator Symbol Classifier | Predictive Analytics Course | "
    "Adithyan | Jia | Theertha"
    "</div>",
    unsafe_allow_html=True
)

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from numpy import argmax
import time

# Page configuration
st.set_page_config(
    page_title="🌿 Plant Disease Recognition",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class indices for prediction
class_indices = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3,
    'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 'Grape___healthy': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20,
    'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27,
    'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30,
    'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32, 'Tomato___Spider_mites Two-spotted_spider_mite': 33,
    'Tomato___Target_Spot': 34, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36,
    'Tomato___healthy': 37
}
class_indices_to_names = {v: k for k, v in class_indices.items()}

@st.cache_resource
def load_trained_model():
    """Load and cache the trained model"""
    return load_model("Plant_Village.h5", compile=False)

def model_prediction(test_image):
    """Predict disease from uploaded image"""
    model = load_trained_model()
    img = image.load_img(test_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    predicted_class = class_indices_to_names[argmax(prediction)]
    return predicted_class, confidence

def format_disease_name(name):
    """Format the disease name for display"""
    parts = name.split('___')
    plant = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else "Unknown"
    return plant, disease

# Sidebar Navigation
with st.sidebar:
    st.title("🌱 Dashboard")
    st.divider()
    
    app_mode = st.radio(
        "📌 Navigate",
        ["🏠 Home", "🔬 Disease Recognition", "ℹ️ About"],
        index=0
    )
    
    st.divider()
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classes", "38")
    with col2:
        st.metric("Plants", "14")
    
    st.divider()
    
    # Supported Plants
    with st.expander("🌿 Supported Plants"):
        st.write("🍎 Apple")
        st.write("🫐 Blueberry")
        st.write("🍒 Cherry")
        st.write("🌽 Corn")
        st.write("🍇 Grape")
        st.write("🍊 Orange")
        st.write("🍑 Peach")
        st.write("🫑 Pepper")
        st.write("🥔 Potato")
        st.write("🍓 Strawberry")
        st.write("🍅 Tomato")

# ==================== HOME PAGE ====================
if app_mode == "🏠 Home":
    st.title("🌿 Plant Disease Recognition System")
    
    # Hero image
    st.image("CMD.jpg", use_container_width=True)
    
    st.divider()
    
    # Welcome Section
    st.header("Welcome! 🚀")
    st.write("""
    Our AI-powered system helps farmers and gardeners identify plant diseases instantly. 
    Upload a photo of your plant leaf and get accurate disease detection in seconds!
    """)
    
    st.divider()
    
    # Features section
    st.header("✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 High Accuracy")
        st.write("State-of-the-art CNN model trained on 54,000+ images")
    
    with col2:
        st.subheader("⚡ Instant Results")
        st.write("Get disease predictions in less than 3 seconds")
    
    with col3:
        st.subheader("🌍 14 Plant Species")
        st.write("Supports major crops including tomatoes, potatoes, and more")
    
    st.divider()
    
    # How it works
    st.header("🔄 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("1️⃣ Upload")
        st.write("Take a clear photo of the affected leaf")
    
    with col2:
        st.subheader("2️⃣ Analyze")
        st.write("Our AI processes your image")
    
    with col3:
        st.subheader("3️⃣ Detect")
        st.write("Disease is identified with confidence score")
    
    with col4:
        st.subheader("4️⃣ Act")
        st.write("Get recommendations for treatment")
    
    st.divider()
    
    # Call to action
    st.info("👉 **Ready to diagnose?** Select **🔬 Disease Recognition** from the sidebar to get started!")

# ==================== DISEASE RECOGNITION PAGE ====================
elif app_mode == "🔬 Disease Recognition":
    st.title("🔬 Disease Recognition")
    st.write("Upload a clear image of the plant leaf you want to analyze")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Disease", type="primary", use_container_width=True):
                with col2:
                    st.subheader("📊 Analysis Results")
                    
                    with st.spinner("🧠 AI is analyzing your image..."):
                        # Progress bar
                        progress = st.progress(0)
                        for i in range(100):
                            time.sleep(0.015)
                            progress.progress(i + 1)
                        
                        result, confidence = model_prediction(uploaded_file)
                        plant, disease = format_disease_name(result)
                    
                    progress.empty()
                    
                    # Results
                    st.success("✅ Analysis Complete!")
                    
                    st.metric("🌱 Plant", plant)
                    st.metric("🔬 Condition", disease)
                    st.metric("📈 Confidence", f"{confidence:.1f}%")
                    
                    st.divider()
                    
                    # Health status
                    is_healthy = "healthy" in disease.lower()
                    
                    if is_healthy:
                        st.balloons()
                        st.success("🎉 **Great news!** Your plant appears to be healthy. Keep up the good care!")
                    else:
                        st.warning(f"⚠️ **Disease detected:** {disease}")
                        st.info("💡 **Recommendation:** Consult with a local agricultural expert for treatment options.")
        else:
            with col2:
                st.subheader("📊 Analysis Results")
                st.info("📷 Upload an image to see results here")

# ==================== ABOUT PAGE ====================
elif app_mode == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.divider()
    
    # Dataset info
    st.header("📊 The Dataset")
    st.write("""
    This project uses the **PlantVillage Dataset**, a comprehensive public dataset containing 
    **54,305 images** of diseased and healthy plant leaves collected under controlled conditions.
    """)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", "54,305")
    with col2:
        st.metric("Crop Species", "14")
    with col3:
        st.metric("Disease Classes", "38")
    with col4:
        st.metric("Basic Diseases", "17")
    
    st.divider()
    
    # Disease categories
    st.header("🦠 Disease Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("🔬 **17** Basic plant diseases")
        st.write("🦠 **4** Bacterial diseases")
        st.write("🍄 **2** Mold-caused diseases")
    
    with col2:
        st.write("🧬 **2** Viral diseases")
        st.write("🕷️ **1** Mite-caused disease")
        st.write("✅ **12** Healthy leaf categories")
    
    st.divider()
    
    # Supported crops
    st.header("🌾 Supported Crops")
    
    crops = ["🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn", "🍇 Grape", 
             "🍊 Orange", "🍑 Peach", "🫑 Pepper", "🥔 Potato", "🍓 Raspberry", 
             "🫘 Soybean", "🎃 Squash", "🍓 Strawberry", "🍅 Tomato"]
    
    cols = st.columns(7)
    for i, crop in enumerate(crops):
        with cols[i % 7]:
            st.write(crop)
    
    st.divider()
    
    # Technology stack
    st.header("🛠️ Technology Stack")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("🐍 Python")
    with col2:
        st.write("🧠 TensorFlow")
    with col3:
        st.write("🎯 Keras")
    with col4:
        st.write("🎨 Streamlit")
    with col5:
        st.write("🐳 Docker")

# Footer
st.divider()
st.caption("Made with 💚 for Agricultural Innovation | © 2024 Plant Disease Recognition System")
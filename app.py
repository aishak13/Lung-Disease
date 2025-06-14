import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image
import io
import base64
from fpdf import FPDF
import tempfile

# Set page config
st.set_page_config(page_title="Lung Cancer Prediction App", layout="wide")

# Load the model
@st.cache_resource
def load_prediction_model():
    return load_model('chest_cancer_model_fine_tuned.h5')

# Define class names
class_names = ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma']

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Function to make prediction
def predict(image):
    model = load_prediction_model()
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0]

# Function to plot confidence scores
def plot_confidence_scores(predictions):
    fig, ax = plt.subplots()
    sns.barplot(x=predictions, y=class_names, ax=ax)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Prediction Confidence Scores')
    return fig

# Function to display cancer information
def display_cancer_info(cancer_type):
    info = {
        "Adenocarcinoma": """
        - Most common type of lung cancer
        - Often found in outer areas of the lung
        - Tends to grow slower than other types
        - Common in both smokers and non-smokers
        """,
        "Large cell carcinoma": """
        - Tends to grow and spread quickly
        - Can appear in any part of the lung
        - Often diagnosed at later stages
        - Accounts for about 10-15% of lung cancers
        """,
        "Squamous cell carcinoma": """
        - Often linked to a history of smoking
        - Usually found in the central part of the lungs
        - Tends to grow slower than other types
        - Accounts for about 25-30% of all lung cancers
        """,
        "Normal": """
        - No signs of cancerous cells
        - Regular lung structure and function
        - Important for early detection and comparison
        """
    }
    return info[cancer_type]

# Function to process multiple images
def process_multiple_images(uploaded_files):
    results = []
    for file in uploaded_files:
        img = Image.open(file)
        predictions = predict(img)
        results.append({
            'filename': file.name,
            'predictions': predictions,
            'predicted_class': class_names[np.argmax(predictions)],
            'image': img
        })
    return results

# Function to compare two images
def compare_images(image1, image2):
    pred1 = predict(image1)
    pred2 = predict(image2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image1)
    ax1.set_title(f"Predicted: {class_names[np.argmax(pred1)]}")
    ax1.axis('off')

    ax2.imshow(image2)
    ax2.set_title(f"Predicted: {class_names[np.argmax(pred2)]}")
    ax2.axis('off')

    plt.tight_layout()
    return fig, pred1, pred2

# Function to generate PDF report
def generate_pdf_report(results):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Lung Cancer Prediction Report", ln=True, align="C")
    pdf.ln(10)

    for result in results:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Analysis for {result['filename']}", ln=True)
        pdf.ln(5)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            result['image'].save(tmpfile, format="PNG")
            pdf.image(tmpfile.name, x=10, w=100)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Predicted Class: {result['predicted_class']}", ln=True)
        pdf.ln(5)

        pdf.cell(0, 10, "Confidence Scores:", ln=True)
        for class_name, confidence in zip(class_names, result['predictions']):
            pdf.cell(0, 10, f"{class_name}: {confidence:.2%}", ln=True)

        pdf.ln(10)

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Information about {result['predicted_class']}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "", 12)
        info = display_cancer_info(result['predicted_class'])
        pdf.multi_cell(0, 10, info)

        pdf.ln(10)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        with open(tmpfile.name, "rb") as f:
            return f.read()

# Main function
def main():
    st.title("Lung Cancer Prediction from CT Scan Images")
    page = st.sidebar.selectbox("Navigate", ["Home"])
    if page == "Home":
        home_page()

def home_page():
    st.write("Upload one or multiple Chest CT Scan images to predict the type of lung cancer.")
    uploaded_files = st.file_uploader("Choose CT Scan image(s)...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        results = process_multiple_images(uploaded_files)

        st.subheader("Analysis Results")
        for result in results:
            st.write(f"**File:** {result['filename']}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(result['image'], caption=f"Uploaded CT Scan: {result['filename']}", use_column_width=True)
            with col2:
                st.write(f"Predicted Class: **{result['predicted_class']}**")
                st.write("Confidence Scores:")
                for class_name, confidence in zip(class_names, result['predictions']):
                    st.write(f"{class_name}: {confidence:.2%}")
                fig = plot_confidence_scores(result['predictions'])
                st.pyplot(fig)

        if len(results) >= 2:
            st.subheader("Comparative Analysis")
            st.write("Select two images to compare:")
            image1 = st.selectbox("Select first image:", [r['filename'] for r in results], key='image1')
            image2 = st.selectbox("Select second image:", [r['filename'] for r in results], key='image2')

            if image1 != image2:
                img1 = next(r['image'] for r in results if r['filename'] == image1)
                img2 = next(r['image'] for r in results if r['filename'] == image2)
                comp_fig, pred1, pred2 = compare_images(img1, img2)
                st.pyplot(comp_fig)

                st.write("Prediction Comparison:")
                for class_name, conf1, conf2 in zip(class_names, pred1, pred2):
                    st.write(f"{class_name}: {conf1:.2%} vs {conf2:.2%}")

                if np.argmax(pred1) != np.argmax(pred2):
                    st.warning("The predictions for these two images differ. Please consult with a healthcare professional for a thorough evaluation.")
                else:
                    st.success("The predictions for these two images are consistent.")
            else:
                st.warning("Please select two different images for comparison.")

        pdf_report = generate_pdf_report(results)
        st.download_button(
            label="Download PDF Report",
            data=pdf_report,
            file_name="lung_cancer_prediction_report.pdf",
            mime="application/pdf"
        )

    st.sidebar.title("Learn About Lung Cancer Types")
    cancer_type = st.sidebar.selectbox("Select a cancer type to learn more:", class_names)
    st.sidebar.write(display_cancer_info(cancer_type))

    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        Disclaimer: This app is for educational purposes only. Always consult with a healthcare professional for medical advice.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

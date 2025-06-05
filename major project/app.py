import os
import numpy as np
import pandas as pd
import pathlib
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
from matplotlib.lines import Line2D
import gradio as gr
import shutil
import base64

# Model Constants
IMAGE_SIZE = (128, 128)
MODEL_FILENAME = "pneumonia_model.h5"

# Check if model exists
def model_exists():
    return os.path.exists(MODEL_FILENAME)

# Data preprocessing function
def preprocess_data(data_paths):
    X = []
    y = []
    
    for key, path in data_paths.items():
        try:
            images = list(pathlib.Path(path).glob('*.jpeg'))
            for img in images:
                image = cv2.imread(str(img))
                if image is not None:
                    resized_img = cv2.resize(image, IMAGE_SIZE)
                    X.append(resized_img)
                    if 'normal' in key.lower():
                        y.append(0)  # Normal
                    else:
                        y.append(1)  # Pneumonia
        except Exception as e:
            print(f"Error processing {key}: {e}")
    
    return np.array(X), np.array(y)

# Define the model architecture
def create_model():
    model = keras.Sequential([
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu',
                    kernel_regularizer=regularizers.l2(0.001),
                    input_shape=IMAGE_SIZE + (3,)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(16, kernel_size=(5, 5), activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),

        layers.Dense(84, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),

        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model_save_path=MODEL_FILENAME):
    # Sample path dictionary - replace with your paths
    # In a real app, let users upload or point to their dataset
    path_dict = {
        'train_data_dir_normal': "chest_xray/train/NORMAL",
        'train_data_dir_pneumonia': "chest_xray/train/PNEUMONIA",
        'test_data_dir_normal': "chest_xray/test/NORMAL",
        'test_data_dir_pneumonia': "chest_xray/test/PNEUMONIA",
        'data_val_dir_normal': "chest_xray/val/NORMAL",
        'data_val_dir_pneumonia': "chest_xray/val/PNEUMONIA",
    }
    
    X, y = preprocess_data(path_dict)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0
    
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    model = create_model()
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        verbose=1,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train_scaled, 
        y_train_cat, 
        epochs=30, 
        validation_data=(X_test_scaled, y_test_cat), 
        callbacks=[early_stopping]
    )
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, history

# SLIC segmentation function
def slic_segmentation(image, n_segments=100):
    """Perform SLIC segmentation on the input image."""
    segments = slic(image, n_segments=n_segments, compactness=10)
    return segments, image

# Calculate Jaccard coefficient between two segmentations
def calculate_jaccard_coefficient(true_segments, lime_segments):
    intersection = np.logical_and(true_segments, lime_segments)
    union = np.logical_or(true_segments, lime_segments)
    jaccard_coefficient = np.sum(intersection) / np.sum(union)
    return jaccard_coefficient

# Function to explain the image using LIME
def explain_image(image, model):
    """Use LIME to explain the model's prediction on the given image."""
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Preprocess the image
    img_resized = cv2.resize(image, IMAGE_SIZE)
    img_processed = img_resized.astype('double') / 255.0
    
    # Get the prediction
    prediction = model.predict(np.expand_dims(img_processed, axis=0))
    pred_class = np.argmax(prediction)
    pred_label = 'Normal' if pred_class == 0 else 'Pneumonia'
    true_label = 'Unknown'  # In a real app, this might come from user input
    
    # Use LIME to explain
    explanation = explainer.explain_instance(
        img_processed, 
        model.predict,
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )
    
    # Get the mask and segmentation
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'Analysis Results: {pred_label} (Confidence: {float(np.max(prediction)):.2f})', 
                fontsize=16, fontweight='bold', color='#2C3E50')
    
    # Use a consistent color scheme
    main_color = '#3498DB'  # Blue for highlights
    support_color = '#2ECC71'  # Green for supporting features
    against_color = '#E74C3C'  # Red for opposing features
    
    # Original Image
    axs[0, 0].imshow(img_processed)
    axs[0, 0].set_title('Original X-Ray Image', fontsize=14, color='#2C3E50')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].spines['bottom'].set_color('#CCCCCC')
    axs[0, 0].spines['left'].set_color('#CCCCCC')
    
    # LIME Mask
    # AFTER (fixed code):
    # LIME Mask - Use a red-green colormap that matches the legend
    # Define a custom colormap from red to green
    custom_cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
    mask_image = axs[0, 1].imshow(mask, cmap=custom_cmap, alpha=0.7)
    axs[0, 1].set_title(f"LIME Heat Map", fontsize=14, color='#2C3E50')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    colorbar = plt.colorbar(mask_image, ax=axs[0, 1], orientation='vertical', shrink=0.8)
    colorbar.set_label('Feature Importance', fontsize=12)
    axs[0, 1].spines['top'].set_visible(False)
    axs[0, 1].spines['right'].set_visible(False)
    axs[0, 1].spines['bottom'].set_color('#CCCCCC')
    axs[0, 1].spines['left'].set_color('#CCCCCC')

    # Add legend with improved styling that matches the actual visualization
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Supports Prediction', 
            markerfacecolor=support_color, markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Against Prediction', 
            markerfacecolor=against_color, markersize=12)
    ]
    leg = axs[0, 1].legend(handles=legend_elements, loc='lower left', framealpha=0.9, 
                        edgecolor='#CCCCCC', facecolor='white', fontsize=12)
    
    # Combined Image with Boundaries
    axs[0, 2].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    axs[0, 2].set_title('Feature Boundaries', fontsize=14, color='#2C3E50')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].spines['top'].set_visible(False)
    axs[0, 2].spines['right'].set_visible(False)
    axs[0, 2].spines['bottom'].set_color('#CCCCCC')
    axs[0, 2].spines['left'].set_color('#CCCCCC')
    
    # Add annotation below the plot
    axs[0, 2].text(0.5, -0.15, 
                  f'Green: Supporting the {pred_label} prediction\nRed: Against the {pred_label} prediction', 
                  transform=axs[0, 2].transAxes, ha='center', va='center', 
                  fontsize=12, color='#7F8C8D', 
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC', boxstyle='round,pad=0.5'))
    
    # Superpixel Regions with Numbers
    superpixel_regions = explanation.segments
    superpixel_img = axs[1, 0].imshow(superpixel_regions, cmap='nipy_spectral', alpha=0.7)
    axs[1, 0].set_title('Superpixel Segmentation', fontsize=14, color='#2C3E50')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].spines['top'].set_visible(False)
    axs[1, 0].spines['right'].set_visible(False)
    axs[1, 0].spines['bottom'].set_color('#CCCCCC')
    axs[1, 0].spines['left'].set_color('#CCCCCC')
    
    # Add segment numbers but limit to keep it cleaner
    unique_segments = np.unique(superpixel_regions)
    if len(unique_segments) > 30:
        # If too many segments, only label some of them
        display_segments = np.random.choice(unique_segments, 30, replace=False)
    else:
        display_segments = unique_segments
        
    for segment_num in display_segments:
        y_coords, x_coords = np.where(superpixel_regions == segment_num)
        if len(y_coords) > 0 and len(x_coords) > 0:
            axs[1, 0].text(np.mean(x_coords), np.mean(y_coords), str(segment_num), 
                          color='white', ha='center', va='center', fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
    
    # Feature Importance Bar Chart with improved styling
    features_normal = [f[0] for f in explanation.local_exp[0] if f[1] > 0]
    weights_normal = [f[1] for f in explanation.local_exp[0] if f[1] > 0]
    
    if 1 in explanation.local_exp:
        features_pneumonia = [f[0] for f in explanation.local_exp[1] if f[1] > 0]
        weights_pneumonia = [f[1] for f in explanation.local_exp[1] if f[1] > 0]
    else:
        features_pneumonia = []
        weights_pneumonia = []
    
    # Sort by weight for better visualization
    if features_normal:
        sorted_normal = sorted(zip(features_normal, weights_normal), key=lambda x: x[1], reverse=True)
        features_normal, weights_normal = zip(*sorted_normal) if sorted_normal else ([], [])
        
    if features_pneumonia:    
        sorted_pneumonia = sorted(zip(features_pneumonia, weights_pneumonia), key=lambda x: x[1], reverse=True)
        features_pneumonia, weights_pneumonia = zip(*sorted_pneumonia) if sorted_pneumonia else ([], [])
    
    # Take only top 10 for clarity
    features_normal = features_normal[:10]
    weights_normal = weights_normal[:10]
    features_pneumonia = features_pneumonia[:10]
    weights_pneumonia = weights_pneumonia[:10]
    
    axs[1, 1].barh(features_normal, weights_normal, color='#3498DB', alpha=0.8, label='Normal', height=0.7)
    axs[1, 1].barh(features_pneumonia, weights_pneumonia, color='#E74C3C', alpha=0.8, label='Pneumonia', height=0.7)
    axs[1, 1].set_xlabel('Importance Score', fontsize=12, color='#2C3E50')
    axs[1, 1].set_title('Top Feature Importance', fontsize=14, color='#2C3E50')
    axs[1, 1].set_yticks(list(features_normal) + list(features_pneumonia))
    axs[1, 1].set_yticklabels([f'Region {i}' for i in list(features_normal) + list(features_pneumonia)], fontsize=10)
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)
    axs[1, 1].spines['bottom'].set_color('#CCCCCC')
    axs[1, 1].spines['left'].set_color('#CCCCCC')
    axs[1, 1].grid(axis='x', linestyle='--', alpha=0.7, color='#CCCCCC')
    leg = axs[1, 1].legend(loc='lower right', fontsize=12, framealpha=0.9, edgecolor='#CCCCCC')
    
    # Prediction confidence visualization
    confidence = float(np.max(prediction))
    colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71']  # Red to green
    
    # Determine color based on confidence
    if confidence < 0.5:
        color_idx = 0
    elif confidence < 0.7:
        color_idx = 1
    elif confidence < 0.9:
        color_idx = 2
    else:
        color_idx = 3
    
    # Create confidence bar
    axs[1, 2].set_visible(True)
    axs[1, 2].set_xlim(0, 1)
    axs[1, 2].set_ylim(0, 1)
    axs[1, 2].barh(0.5, confidence, height=0.3, color=colors[color_idx], alpha=0.8)
    axs[1, 2].barh(0.5, 1, height=0.3, color='#CCCCCC', alpha=0.3)
    axs[1, 2].set_title('Prediction Confidence', fontsize=14, color='#2C3E50')
    axs[1, 2].text(confidence/2, 0.5, f'{confidence:.2f}', ha='center', va='center', color='white', fontweight='bold')
    axs[1, 2].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[1, 2].set_yticks([])
    axs[1, 2].spines['top'].set_visible(False)
    axs[1, 2].spines['right'].set_visible(False)
    axs[1, 2].spines['left'].set_visible(False)
    
    # Add confidence level indicator
    confidence_text = 'Low'
    if confidence >= 0.9:
        confidence_text = 'Very High'
    elif confidence >= 0.7:
        confidence_text = 'High'
    elif confidence >= 0.5:
        confidence_text = 'Medium'
        
    axs[1, 2].text(0.5, 0.1, f'Confidence Level: {confidence_text}', ha='center', va='center', 
                  transform=axs[1, 2].transAxes, fontsize=12, color='#2C3E50',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC', boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    return fig, pred_label, confidence

# Gradio interface functions
def analyze_xray(input_img):
    """Function called by Gradio when analyzing an X-ray image."""
    if input_img is None:
        return None, "⚠️ Please upload an image"
    
    # Load model (in a real app, load this once at startup)
    if model_exists():
        model = keras.models.load_model(MODEL_FILENAME)
    else:
        return None, "⚠️ Model not found. Please train the model first."
    
    # Ensure image is in RGB format
    if len(input_img.shape) == 2:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
    elif input_img.shape[2] == 4:  # RGBA
        input_img = input_img[:, :, :3]  # Convert to RGB
    
    try:
        # Generate the explanation visualization
        fig, pred_label, confidence = explain_image(input_img, model)
        
        # Return the figure with improved result message
        confidence_color = "#E74C3C"  # Red for low confidence
        if confidence >= 0.9:
            confidence_color = "#2ECC71"  # Green for high confidence
        elif confidence >= 0.7:
            confidence_color = "#F1C40F"  # Yellow for medium confidence
            
        result_message = f"""
        ## 📊 Analysis Results
        
        **Prediction:** {pred_label}
        
        **Confidence:** <span style="color:{confidence_color}; font-weight:bold;">{confidence:.2f}</span>
        
        *AI assisted diagnosis should always be verified by a medical professional.*
        """
        
        return fig, result_message
        
    except Exception as e:
        return None, f"⚠️ Error: {str(e)}"

def train_model_callback():
    """Function called by Gradio to train the model."""
    if model_exists():
        return """
        ### ⚠️ Model already exists
        
        Delete the existing model file to retrain.
        """
    
    try:
        model, history = train_model()
        return f"""
        ### ✅ Model trained successfully!
        
        Final accuracy: **{history.history['accuracy'][-1]:.4f}**
        
        The model is now ready to analyze X-ray images.
        """
    except Exception as e:
        return f"""
        ### ❌ Error training model
        
        {str(e)}
        
        Please check that your dataset is properly configured.
        """

# Function to create the necessary favicon files in Gradio's temp directory
def setup_favicon():
    # Create temp directory for favicon if it doesn't exist
    os.makedirs('favicon_files', exist_ok=True)
    
    # Copy the favicon.png to the temp directory
    if os.path.exists('favicon.png'):
        shutil.copy('favicon.png', 'favicon_files/favicon.png')
        print("Favicon copied to temp directory")
        return True
    else:
        print("Favicon.png not found in the current directory")
        return False
def validate_and_analyze(input_img):
    """Validate image before analyzing"""
    if input_img is None:
        return None, "⚠️ Please upload an image"
    
    # Calculate image size in memory
    img_size_bytes = input_img.nbytes
    max_size_bytes = 5 * 1024 * 1024  # 5MB
    
    if img_size_bytes > max_size_bytes:
        return None, f"⚠️ Image too large ({img_size_bytes / (1024 * 1024):.2f}MB). Maximum size is 5MB."
    
    # Continue with normal analysis
    return analyze_xray(input_img)

# Custom HTML with embedded favicon
def get_favicon_html():
    if os.path.exists('favicon.png'):
        try:
            # Read the favicon file as binary
            with open('favicon.png', 'rb') as f:
                favicon_data = f.read()
                # Convert to base64
                favicon_b64 = base64.b64encode(favicon_data).decode('utf-8')
                
                # Create HTML with embedded favicon
                return f"""
                <script>
                    // Create a favicon link element
                    var link = document.createElement('link');
                    link.rel = 'icon';
                    link.type = 'image/png';
                    link.href = 'data:image/png;base64,{favicon_b64}';
                    
                    // Get the existing favicon
                    var oldLink = document.querySelector('link[rel="icon"]');
                    if (oldLink) {{
                        document.head.removeChild(oldLink);
                    }}
                    
                    // Add the new favicon to the head
                    document.head.appendChild(link);
                </script>
                """
        except Exception as e:
            print(f"Error creating favicon HTML: {e}")
            return ""
    else:
        return ""

custom_css = """
/* Hide footer elements */
footer {
    display: none !important;
}

/* General UI improvements */
body {
    background-color: #F5F7FA !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding-top: 20px !important;
}

h1 {
    color: #f4f4f5 !important;
    font-size: 2.5rem !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
    font-weight: 700 !important;
}

h2 {
    color: #34495E !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    color: #2980B9 !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}

.gradio-button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.gradio-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
}

/* Custom styles for primary/secondary buttons */
.gradio-button.primary {
    background-color: #3498DB !important;
    border: none !important;
}

.gradio-button.primary:hover {
    background-color: #2980B9 !important;
}

.gradio-button.secondary {
    background-color: #95A5A6 !important;
    border: none !important;
}

.gradio-button.secondary:hover {
    background-color: #7F8C8D !important;
}

/* Result area styling */
.output-markdown p {
    font-size: 1.1rem !important;
    line-height: 1.5 !important;
}

/* Card-like containers */
.gradio-box {
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    background-color: white !important;
    border: 1px solid #E0E0E0 !important;
    overflow: hidden !important;
}

/* Image upload area */
.upload-box {
    border: 2px dashed #3498DB !important;
    border-radius: 10px !important;
    background-color: rgba(52, 152, 219, 0.05) !important;
    transition: all 0.3s ease !important;
}

.upload-box:hover {
    border-color: #2980B9 !important;
    background-color: rgba(52, 152, 219, 0.1) !important;
}

/* Plot area */
.plot-container {
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* App header/subtitle */
.app-subtitle {
    text-align: center !important;
    color: #7F8C8D !important;
    margin-bottom: 2rem !important;
    font-size: 1.2rem !important;
}

/* Instructions styling */
.instructions-box {
    background-color: #1f1f1f !important;
    # border-left: 4px solid #3498DB !important;
    padding: 12px !important;
    margin: 1rem 0 !important;
    border-radius: 0 8px 8px 0 !important;
}

.instructions-title {
    font-weight: 600 !important;
    color: #2C3E50 !important;
    margin-bottom: 6px !important;
}

.instructions-step {
    font-weight: normal !important;
    margin-bottom: 4px !important;
}
"""

# ========================
# Main Application
# ========================

# Create Gradio interface
with gr.Blocks(title="Pneumonia X-Ray Analysis", css=custom_css) as demo:
    # Add favicon via HTML injection at the top of the page
    favicon_html = get_favicon_html()
    if favicon_html:
        gr.HTML(favicon_html)
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 1rem">
        <h1>🫁 Pneumonia Detection from Chest X-Rays</h1>
        <p class="app-subtitle">
            AI-powered analysis with transparent explanations using LIME technology
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background-color:#1f1f1f; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                <h2 style="margin-top: 0; color: #E8F4FC;">📋 How It Works</h2>
                <p>This tool uses deep learning to analyze chest X-rays and detect signs of pneumonia, 
                while using LIME (Local Interpretable Model-agnostic Explanations) to explain the AI's decisions.</p>
            </div>
            """)
            
            # With this updated code that adds file type and size restrictions:
            with gr.Group(elem_classes="upload-box"):
                input_image = gr.Image(
                    elem_id="input-image",
                    sources=["upload", "webcam", "clipboard"]
                )
                

            
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze X-ray", variant="primary", elem_id="analyze-btn")
                train_btn = gr.Button("🧠 Train Model", variant="secondary", elem_id="train-btn")
            
            output_text = gr.Markdown(label="Result", elem_id="result-text")
            
            gr.HTML("""
            <div class="instructions-box">
                <div class="instructions-title">📝 Instructions:</div>
                <ol>
                    <li class="instructions-step">If running for the first time, click "Train Model" (requires dataset)</li>
                    <li class="instructions-step">Upload a chest X-ray image</li>
                    <li class="instructions-step">Click "Analyze X-ray" to get the prediction and explanation</li>
                </ol>
            </div>
            """)
        
        with gr.Column(scale=2):
            with gr.Group():
                output_plot = gr.Plot(label="AI Explanation Visualization", elem_id="output-plot")
            
            gr.HTML("""
            <div style="background-color: #1f1f1f; border-radius: 10px; padding: 15px; margin-top: 20px;">
                <h2 style="margin-top: 0; color: #34495E;">🔍 Understanding the Visualization</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <h3>Original Image</h3>
                        <p>The chest X-ray that was uploaded for analysis</p>
                        
                        <h3>LIME Heat Map</h3>
                        <p>Highlights which regions the model is focusing on for its prediction, with brighter areas indicating higher importance</p>
                    </div>
                    <div>
                        <h3>Feature Boundaries</h3>
                        <p>Green regions support the prediction, red regions oppose it</p>
                        
                        <h3>Superpixel Segmentation</h3>
                        <p>Shows how the image is divided into regions for analysis</p>
                    </div>
                </div>
                <p style="margin-top: 10px; color: #7F8C8D; font-style: italic;">
                    This tool is for educational purposes only. All medical decisions should be made by qualified healthcare professionals.
                </p>
            </div>
            """)
    
    analyze_btn.click(
        fn=validate_and_analyze,
        inputs=[input_image],
        outputs=[output_plot, output_text]
    )
    
    train_btn.click(
        fn=train_model_callback,
        inputs=[],
        outputs=[output_text]
    )

# For direct execution
if __name__ == "__main__":
    # Setup favicon
    setup_favicon()
    
    # Check if model exists, if not, ask if user wants to train
    if not model_exists():
        print("Model not found. Training is required before analyzing images.")
        train_choice = input("Do you want to train the model now? (y/n): ")
        if train_choice.lower() == 'y':
            model, _ = train_model()
    
    # Try multiple approaches for the favicon
    try:
        # First approach: standard favicon_path
        demo.launch(favicon_path="favicon.png")
    except Exception as e1:
        try:
            # Second approach: try with absolute path
            favicon_abs_path = os.path.abspath("favicon.png")
            demo.launch(favicon_path=favicon_abs_path)
        except Exception as e2:
            # Fallback: launch without favicon parameter
            print(f"Could not set favicon: {e1}, {e2}")
            print("Launching with embedded favicon in HTML")
            
            demo.launch()
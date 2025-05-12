import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image

# Load models (you can add more models here)
MODELS = {
    'lenet': 'models/PneumoniaLENET_LIME.h5',
    # Add paths to your other models
}

def load_pretrained_model(model_name='lenet'):
    """Load the pre-trained model"""
    model_path = MODELS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"Model {model_name} not found at {model_path}")
    return load_model(model_path)

def preprocess_image(img_path, target_size=(150, 150)):
    """Preprocess the image for prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

def predict_pneumonia(img_path, model_name='lenet'):
    """Make a prediction on the image"""
    model = load_pretrained_model(model_name)
    img_array = preprocess_image(img_path)
    
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Pneumonia", float(prediction[0][0])
    else:
        return "Normal", 1 - float(prediction[0][0])

def generate_lime_explanation(img_path, model_name='lenet', save_path='static/lime_explanations'):
    """Generate LIME explanation for the prediction"""
    os.makedirs(save_path, exist_ok=True)
    model = load_pretrained_model(model_name)
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Preprocess image for LIME
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    
    # Explain the prediction
    explanation = explainer.explain_instance(
        img_array, 
        model.predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )
    
    # Get explanation image
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    # Create and save explanation image
    explanation_img = mark_boundaries(temp / 2 + 0.5, mask)
    explanation_img = (explanation_img * 255).astype(np.uint8)
    
    # Generate output filename
    base_name = os.path.basename(img_path).split('.')[0]
    output_path = os.path.join(save_path, f"{base_name}_lime_{model_name}.png")
    
    # Save the image
    plt.imsave(output_path, explanation_img)
    
    return output_path
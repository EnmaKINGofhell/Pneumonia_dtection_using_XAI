
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from lime_explainer import predict_pneumonia, generate_lime_explanation

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get model selection
        model_name = request.form.get('model', 'lenet')
        
        # Get prediction and LIME explanation
        prediction, confidence = predict_pneumonia(filepath, model_name)
        lime_img_path = generate_lime_explanation(filepath, model_name)
        
        return render_template('results.html', 
                             uploaded_image=filepath,
                             prediction=prediction,
                             confidence=confidence,
                             lime_explanation=lime_img_path,
                             model_used=model_name.upper())
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
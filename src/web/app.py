import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
from pathlib import Path
import sys

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.gradcam import apply_gradcam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload directory if it doesn't exist
os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)

# Load the trained model
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
model_path = project_root / 'models' / 'pneumonia_model_final.h5'

try:
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train the model first.")
        model = None
    else:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img = img / 255.0
    
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    # For ResNet50, we need to convert grayscale to 3 channels
    if model.input_shape[-1] == 3:
        img = np.concatenate([img, img, img], axis=-1)
    
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess the image
            img = preprocess_image(file_path)
            
            # Make prediction
            prediction = model.predict(img)[0][0]
            label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
            probability = float(prediction) if prediction > 0.5 else float(1 - prediction)
            
            # Generate Grad-CAM visualization
            try:
                _, _, superimposed_img, _, _ = apply_gradcam(
                    file_path, model, last_conv_layer_name="conv5_block3_out"
                )
                
                # Save the visualization
                vis_filename = f"gradcam_{filename}"
                vis_path = os.path.join(app.config['UPLOAD_FOLDER'], vis_filename)
                cv2.imwrite(os.path.join(app.root_path, vis_path), 
                           cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
                
                return jsonify({
                    'success': True,
                    'label': label,
                    'probability': probability,
                    'image_path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
                    'visualization_path': vis_path
                })
            except Exception as e:
                print(f"Visualization error: {str(e)}")
                return jsonify({
                    'success': True,
                    'label': label,
                    'probability': probability,
                    'image_path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
                    'error': f"Could not generate visualization: {str(e)}"
                })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f"Error processing image: {str(e)}"
            })
    
    return jsonify({'error': 'Invalid file format. Please upload a PNG, JPG, or JPEG image.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Create a Grad-CAM heatmap for model visualization.
    
    Args:
        img_array: Input image as a numpy array
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer in the model
        pred_index: Index of the class to visualize (None for the highest scoring class)
        
    Returns:
        heatmap: Grad-CAM heatmap
    """
    try:
        # First try to get the layer by name
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # If layer name not found, try to find the last convolutional layer
        conv_layers = [layer.name for layer in model.layers 
                      if 'conv' in layer.name.lower() or 
                         isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("Could not find any convolutional layer in the model")
        last_conv_layer_name = conv_layers[-1]
        last_conv_layer = model.get_layer(last_conv_layer_name)
        print(f"Using {last_conv_layer_name} as the last convolutional layer")
    
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the output neuron with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam(image_path, model, last_conv_layer_name="conv5_block3_out", alpha=0.4):
    """
    Apply Grad-CAM to an image and return the visualization.
    
    Args:
        image_path: Path to the input image
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer in the model
        alpha: Transparency factor for the heatmap overlay
        
    Returns:
        Original image, heatmap, and superimposed visualization
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    
    # Create a copy for visualization
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Prepare the image for the model
    img_array = np.expand_dims(img, axis=0) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    # For ResNet50, we need to convert grayscale to 3 channels
    if model.input_shape[-1] == 3:
        img_tensor = np.concatenate([img_array, img_array, img_array], axis=-1)
    else:
        img_tensor = img_array
    
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name)
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize heatmap to match the input image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img_display * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    # Make prediction
    prediction = model.predict(img_tensor)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    probability = prediction if prediction > 0.5 else 1 - prediction
    
    return img_display, heatmap, superimposed_img, label, probability

def visualize_gradcam(image_path, model, output_path=None, last_conv_layer_name="conv5_block3_out"):
    """
    Visualize the Grad-CAM results for an image and save the visualization.
    
    Args:
        image_path: Path to the input image
        model: Trained Keras model
        output_path: Path to save the visualization (optional)
        last_conv_layer_name: Name of the last convolutional layer in the model
    """
    # Apply Grad-CAM
    original, heatmap, superimposed, label, probability = apply_gradcam(
        image_path, model, last_conv_layer_name
    )
    
    # Create the visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Prediction: {label} ({probability:.2%})")
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    return plt
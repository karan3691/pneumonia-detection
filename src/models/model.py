import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_shape=(224, 224, 1), learning_rate=0.0001):
    """
    Create a CNN model for pneumonia detection using transfer learning with ResNet-50.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        learning_rate (float): Learning rate for the optimizer
        
    Returns:
        model: Compiled Keras model
    """
    # Handle grayscale images by duplicating the channel to make it compatible with ResNet50
    if input_shape[2] == 1:
        inputs = tf.keras.Input(shape=input_shape)
        # Convert grayscale to RGB by duplicating the channel
        x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
    else:
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
    
    # Load pre-trained ResNet50 model without the top layer
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=x)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers for binary classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification (normal vs pneumonia)
    
    # Create the model
    model = Model(inputs=inputs, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def fine_tune_model(model, num_layers_to_unfreeze=10):
    """
    Fine-tune the model by unfreezing some of the top layers of the base model.
    
    Args:
        model: The pre-trained model
        num_layers_to_unfreeze: Number of layers to unfreeze from the top
        
    Returns:
        model: Fine-tuned model
    """
    # Find the ResNet50 base model within our model
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # This is our base_model
            base_model = layer
            break
    
    # Unfreeze the top layers of the base model
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model
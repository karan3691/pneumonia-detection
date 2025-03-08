import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns

# Import our model architecture
from model import create_model, fine_tune_model

def train_model(data_dir=None, img_size=(224, 224), batch_size=32, epochs=20, fine_tune_epochs=10):
    """
    Train the pneumonia detection model using the preprocessed data.
    
    Args:
        data_dir (str): Path to the processed data directory
        img_size (tuple): Input image size (height, width)
        batch_size (int): Batch size for training
        epochs (int): Number of epochs for initial training
        fine_tune_epochs (int): Number of epochs for fine-tuning
    
    Returns:
        model: Trained Keras model
        history: Training history
    """
    # Define paths using absolute paths based on the current file location
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    
    if data_dir is None:
        data_dir = project_root / 'data' / 'processed'
    else:
        data_dir = Path(data_dir)
    
    # Check if processed data exists
    if not data_dir.exists():
        print(f"Processed data directory not found at {data_dir}")
        print("Please run preprocess.py first.")
        return None, None
    
    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased rotation range
        width_shift_range=0.15,  # Increased shift range
        height_shift_range=0.15,  # Increased shift range
        shear_range=0.15,  # Increased shear range
        zoom_range=0.2,  # Increased zoom range
        horizontal_flip=True,
        vertical_flip=False,  # X-rays shouldn't be flipped vertically
        brightness_range=[0.8, 1.2],  # Add brightness variation
        fill_mode='nearest'
        # class_mode is not a parameter for ImageDataGenerator, it belongs in flow_from_directory
    )
    
    # Just rescaling for validation and test data
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir / 'train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale'
    )
    
    # Load validation data
    validation_generator = val_test_datagen.flow_from_directory(
        data_dir / 'val',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale'
    )
    
    # Load test data
    test_generator = val_test_datagen.flow_from_directory(
        data_dir / 'test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
    )
    
    # Create model
    model = create_model(input_shape=(img_size[0], img_size[1], 1))
    
    # Define callbacks
    checkpoint_dir = project_root / 'models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        checkpoint_dir / 'pneumonia_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks
    )
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    model = fine_tune_model(model)
    
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks
    )
    
    # Combine histories
    # Handle potential metric name differences between initial training and fine-tuning
    for key in list(history.history.keys()):
        # Find matching keys in fine_tune_history, accounting for potential suffix differences
        if key in fine_tune_history.history:
            history.history[key].extend(fine_tune_history.history[key])
        else:
            # Look for keys with suffixes (e.g., 'precision' vs 'precision_1')
            base_key = key.split('_')[0] if '_' in key else key
            matching_keys = [k for k in fine_tune_history.history.keys() 
                            if k.startswith(base_key)]
            
            if matching_keys:
                # Use the first matching key
                history.history[key].extend(fine_tune_history.history[matching_keys[0]])
                print(f"Matched '{key}' with '{matching_keys[0]}' in fine-tuning history.")
            else:
                print(f"Warning: Could not find matching key for '{key}' in fine-tuning history.")
    
    # Save the final model
    model.save(checkpoint_dir / 'pneumonia_model_final.h5')
    print(f"Model saved to {checkpoint_dir / 'pneumonia_model_final.h5'}")
    
    # Evaluate the model on test data
    evaluate_model(model, test_generator, checkpoint_dir)
    
    return model, history

def evaluate_model(model, test_generator, output_dir):
    """
    Evaluate the trained model on the test set and generate performance metrics.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        output_dir: Directory to save evaluation results
    """
    print("\nEvaluating the model on test data...")
    
    # Get predictions
    test_generator.reset()
    y_pred_prob = model.predict(test_generator, steps=len(test_generator), verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = test_generator.classes
    
    # Calculate metrics
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator, steps=len(test_generator))
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Generate classification report
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / 'confusion_matrix.png')
    
    # Save metrics to a text file
    with open(output_dir / 'evaluation_metrics.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Evaluation metrics saved to {output_dir / 'evaluation_metrics.txt'}")

def plot_training_history(history, output_dir=None):
    """
    Plot the training history and save the plots.
    
    Args:
        history: Training history object
        output_dir: Directory to save plots
    """
    # Define paths using absolute paths if output_dir is not provided
    if output_dir is None:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        output_dir = project_root / 'models'
    else:
        output_dir = Path(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png')
    print(f"Training history plot saved to {output_dir / 'training_history.png'}")

if __name__ == "__main__":
    model, history = train_model()
    if history is not None:
        plot_training_history(history)
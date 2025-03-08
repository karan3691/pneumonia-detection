# Pneumonia Detection from Chest X-Ray Images

This project implements a deep learning model (CNN) to detect pneumonia from chest X-ray images. The model is deployed as a web application where users can upload X-ray images and receive predictions.

## Project Structure

```
pneumonia/
├── data/                  # Data directory
│   ├── raw/               # Raw downloaded dataset
│   └── processed/         # Processed images
├── models/                # Saved model files
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── src/                   # Source code
│   ├── data/              # Data processing scripts
│   ├── models/            # Model architecture and training
│   ├── visualization/     # Visualization utilities (Grad-CAM)
│   └── web/               # Web application
├── requirements.txt       # Project dependencies
├── setup.py               # Package setup
└── README.md              # Project documentation
```

## Features

- Deep learning model (CNN) achieving 92.5% test accuracy
- Transfer learning with pre-trained ResNet-50 architecture
- Interactive web interface with Grad-CAM visualizations
- Comprehensive data augmentation pipeline
- Model evaluation with accuracy, precision, recall, and F1-score metrics
- Web application for real-time predictions
- Grad-CAM visualizations for model interpretability

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia) dataset from Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), which contains thousands of labeled X-ray images categorized as normal and pneumonia.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
python src/data/download_data.py  # Download the dataset from Kaggle
python src/data/preprocess.py      # Preprocess the images
```

### Model Training

```bash
python src/models/train.py  # Train the model
```

### Web Application

1. Start the Flask server:
```bash
python src/web/app.py
```
2. Open http://localhost:5000 in your browser
3. Upload a chest X-ray image for instant prediction
4. View Grad-CAM visualizations highlighting areas of interest

![Grad-CAM Visualization](static/static.jpeg)
*Heatmap showing model focus areas for pneumonia detection*

## Model Performance

| Metric        | Score   |
|---------------|---------|
| Accuracy      | 92.5%   |
| Precision     | 94.1%   |
| Recall        | 91.8%   |
| F1-score      | 92.9%   |

## Deployment

The web application will soon deployed on [Heroku/AWS/Google Cloud] and can be accessed.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning
- [Flask](https://flask.palletsprojects.com/) for web application
- [Grad-CAM](https://arxiv.org/abs/1610.02391) for visualization
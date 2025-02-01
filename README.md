# Heart Disease Prediction using Gradio

## Overview
This project predicts the likelihood of heart disease based on user inputs. It utilizes machine learning to analyze patient data and provides an easy-to-use Gradio web interface for predictions.

## Features
- User-friendly web interface powered by Gradio
- Machine learning model trained on heart disease dataset
- Real-time predictions based on user inputs
- Scikit-learn model serialization with `.pkl` files

## Technologies Used
- Python
- Gradio
- Scikit-learn
- Pandas & NumPy
- Jupyter Notebook

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Heart_Disease_Prediction-GRADIO.git
   cd Heart_Disease_Prediction-GRADIO
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Gradio interface:
   ```bash
   python heard_gradio.py
   ```
4. Open the provided local link in your browser.

## Usage
- Run the `heard_gradio.py` script to start the web interface.
- Enter patient health parameters in the input fields.
- Click 'Predict' to see the likelihood of heart disease.

## Dataset
The project uses `heart.csv`, which contains patient data with various health parameters. The machine learning model is trained on this dataset.

## Model
- A trained machine learning model (`heart_failure.pkl`) is used for predictions.
- A scaler (`scaler.pkl`) is applied to preprocess the input data before making predictions.

## Contributors
- Your Name ([@yourgithub](https://github.com/anfas-kp))

## License
This project is licensed under the MIT License.


# Revitsone Behavior Detection

## Introduction

Revitsone Behavior Detection is a project aimed at detecting the behavior of drivers while driving vehicles using machine learning techniques. This repository contains the necessary files and code for training models, making predictions, and deploying a Streamlit web application for real-time behavior detection.

### Installation

To use this project, follow these steps to set up your environment:

1. **Clone the Repository**: Clone this repository to your local machine.

2. **Create Virtual Environment**: Navigate to the project directory and create a virtual environment.
   ```bash
   cd project-directory
   python3 -m venv venv


## Environment Setup

| Step                  | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| Activate Virtual Environment | Activate the virtual environment using: `source venv/bin/activate` |
| Install Dependencies  | Install required dependencies using: `pip install -r requirements.txt` |

## Usage

### Model Training

#### Data Preparation

Download the dataset 'Revitsone-5classes' from the directory.

#### Training the Model

Use `model.py` to train the model. Adjust the code and parameters as needed for your dataset and desired model architecture.

### Model Evaluation and Prediction

#### Model Evaluation

After training, evaluate the performance of the model using `evaluation.py`. This will provide insights into the model's accuracy and other metrics.

#### Prediction

Use `prediction.py` to make predictions on new data or in real-time scenarios. This script will utilize the trained model to classify driver behavior.

### Streamlit App

#### Running the App

Execute `app.py` to launch the Streamlit web application. This app provides a user-friendly interface for behavior detection, allowing users to upload images and receive predictions.

### Additional Notes

- Ensure that you have the necessary data stored in the appropriate directories and adjust file paths and configurations as needed for your environment.
- Customize the code and functionality to suit your specific requirements or add additional features.
- For any questions or issues, refer to the documentation within the code or reach out for support.

## Repository Structure

| File/Directory        | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| README.md             | This file, providing an overview and instructions for the project. |
| app.py                | Streamlit web application code for behavior detection.         |
| img_928.jpg           | Sample image for testing purposes.                             |
| inception_model.h5    | Pre-trained model weights.                    |
| model.py              | Code for training machine learning model.                     |
| prediction.py         | Script for making predictions using trained model.            |
| requirements.txt      | List of required Python modules for installation.              |

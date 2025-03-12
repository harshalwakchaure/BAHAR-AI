Plant Disease Recognition System

Overview

This project is an AI-powered plant disease detection system that utilizes deep learning (CNN) and Streamlit to classify plant diseases from leaf images. The model is trained on a dataset of 87,000 images across 38 plant disease categories, achieving 87% accuracy.

Features

Deep Learning Model: Uses a Convolutional Neural Network (CNN) for disease classification.

Dataset: Trained on 87K labeled images of healthy and diseased leaves.

Real-Time Prediction: Upload plant images to detect diseases instantly.

Web Interface: Streamlit-powered UI for easy interaction.

Optimized Performance: Fine-tuned model with data augmentation and preprocessing.

Tech Stack

Machine Learning: TensorFlow, Keras

Data Processing: OpenCV, NumPy, Pandas

Web App: Streamlit

Visualization: Matplotlib, Seaborn

Installation & Setup

Clone the Repository

git clone https://github.com/your-username/Plant-Disease-Recognition.git
cd Plant-Disease-Recognition

Install Dependencies

pip install -r requirements.txt

Run the Web App

streamlit run main.py

Upload a Plant Image and view the disease classification results.

Usage Guide

Open the Disease Recognition page.

Upload an image of a plant leaf.

Click Predict to analyze the image.

View the detected disease and suggested solutions.

Model Training

Used CNN architecture with Conv2D, MaxPooling2D, Flatten, and Dense layers.

Applied data augmentation (rotation, flipping, zooming) to improve accuracy.

Achieved 87% accuracy after 10 epochs.

Project Structure

📂 Plant-Disease-Recognition
│── main.py                 # Streamlit web app
│── Train_plant_disease.ipynb  # Model training script
│── Test_plant_disease.ipynb   # Model testing script
│── trained_plant_disease_model.keras  # Trained model file
│── requirements.txt         # Project dependencies
│── training_hist.json       # Training history
└── README.md               # Project documentation

Future Enhancements

Improve model accuracy with Transfer Learning.

Add a disease treatment recommendation system.

Deploy the model as a cloud-based API.

Contributors

Harshal Wakchaure

Feel free to contribute to this project!

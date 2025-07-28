# Veracity Vigilance: Fake News Detector

Veracity Vigilance is a web application that uses a state-of-the-art AI model to detect fake news. It provides a simple interface for users to paste news articles and receive an instant classification of "REAL" or "FAKE" along with a confidence score, helping to combat the spread of misinformation online.

## Features

Real-Time Analysis: Get instant predictions for any news text you provide.

High Accuracy: Powered by a fine-tuned Transformer model for nuanced understanding of language context.

Confidence Scoring: Each prediction is accompanied by a confidence score, indicating the model's certainty.

CPU-Optimized: Designed to run efficiently on standard machines without requiring a dedicated GPU.

User-Friendly Web Interface: A clean and simple web page built with Flask for easy interaction.

## Technology Stack

This project is built with a modern stack for machine learning and web development:

Backend: Python, Flask

Machine Learning: PyTorch, Hugging Face Transformers

AI Model: microsoft/MiniLM-L12-H384-uncased (a compact and powerful BERT-based model)

Data Processing: Pandas, Scikit-learn

Frontend: HTML, CSS

## Setup and Installation

It's highly recommended to use a virtual environment to manage project dependencies.

## How to Use

Running the application is a two-step process.

Step 1: Train the Model
First, you need to train the AI model on the provided dataset. This script will process the data, fine-tune the MiniLM model, and save the final, trained version to the models/saved_models/ directory.

Note: This process is computationally intensive and will take a significant amount of time (potentially 30 minutes to a few hours) depending on your CPU. You only need to do this once.

python train_model.py

Step 2: Run the Web Application
Once the model is trained and saved, you can start the Flask web server at any time.

python src/app.py

The terminal will show that the server is running. Now, open your web browser and navigate to:

http://127.0.0.1:5000

You can now use the application to analyze news articles.
## Only News Articles From Dataset can be used in the Project 

## Model Details

The core of this project is a fine-tuned MiniLM model. This model was chosen for several key reasons:

Efficiency: It is a distilled version of a larger BERT model, offering a great balance between high performance and computational efficiency.

CPU Performance: It is specifically designed to run well on CPUs, making the project accessible to users without expensive GPU hardware.

Contextual Understanding: As a Transformer-based model, it excels at understanding the context and nuances of language, which is critical for distinguishing between factual and fabricated news.

The model was fine-tuned on the "Fake and real news dataset" from Kaggle, which provides a large and topically diverse collection of articles, ensuring the model learns robust patterns rather than simple topic-based biases.

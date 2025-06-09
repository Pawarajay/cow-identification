Cow Identification and Health prediction System

 Cow Identification and Health Monitoring System

This project identifies individual cows based on images and predicts their eye health status (e.g., Normal or Conjunctivitis) using trained Machine Learning models. The predictions are integrated into a Flask-based web application and can also be used in an Android app via TensorFlow Lite.

---

## 📌 Project Overview

The goal is to help farmers and veterinarians:

- Automatically *identify cows* using image-based recognition.
- Detect *eye diseases* such as *Conjunctivitis*.
- Provide real-time feedback using an *Android App* or a *Web Application*.

---

## 🛠 Tech Stack

- *Frontend:* HTML, CSS (via Flask templates)
- *Backend:* Python (Flask)
- *ML Models:* TensorFlow, TFLite (converted from CNN models)
- *Android Integration:* TensorFlow Lite (.tflite) models
- *Model Input Size:* 224 x 224 pixels

---

## 🧠 Machine Learning Models Used

1. *Cow Identification Model*
   - Type: Convolutional Neural Network (CNN)
   - Output: Cow ID (e.g., Cow1, Cow2, Cow3)
   - Format: .tflite

2. *Eye Health Classification Model*
   - Type: CNN
   - Output: "Normal" or "Conjunctivitis"
   - Format: .tflite

---

## 🌐 Features

- Upload a cow image via web UI.
- Predict:
  - Cow ID (Who is the cow?)
  - Health Status (Is the eye healthy?)
- Displays prediction results on the same page.
- Ready for Android App Integration.

---

## 📁 Folder Structure

project/
│
├── static/
│ └── uploads/ # Folder to store uploaded images
│
├── templates/
│ └── index.html # Main HTML template
│
├── cow_id.tflite # Cow Identification Model
├── cow_eye.tflite # Health Detection Model
├── test.py # Flask application file
├── README.md # You are here!

2. Install Required Packages
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

Install dependencies:

bash
Copy code
pip install -r requirements.txt
If requirements.txt doesn't exist, manually install:

bash
Copy code
pip install flask tensorflow pillow


3. Place the Models
Place your .tflite models in the project root:

cow_id.tflite

cow_eye.tflite

Make sure paths in test.py match their actual locations.

4. Run the Flask App
bash
Copy code
python test.py
Visit the app at: http://127.0.0.1:5000

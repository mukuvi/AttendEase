from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import psycopg2

app = Flask(__name__)

# Connect to PostgreSQL
def connect_db():
    conn = psycopg2.connect(
        dbname="attendance_system", 
        user="your_user", 
        password="your_password", 
        host="localhost", 
        port="5432"
    )
    return conn

# Facial Recognition model loading
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('models/facial_recognition_model.h5')

# Student Registration Endpoint
@app.route('/register', methods=['POST'])
def register_student():
    data = request.json
    name = data['name']
    admission_number = data['admission_number']
    email = data['email']
    department = data['department']
    # Insert student data into PostgreSQL database
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO students (name, admission_number, email, department) 
        VALUES (%s, %s, %s, %s)
    """, (name, admission_number, email, department))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"message": "Student registered successfully"})

# Facial Recognition and Attendance Marking Endpoint
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    face_image = np.array(data['image'])
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = face_image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))  # Resize for model input
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        prediction = model.predict(face)  # Model inference

        # Mark attendance based on face recognition result (details depend on your model output)
        student_id = prediction.argmax()  # Assuming the model gives a class index
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("UPDATE students SET attendance = TRUE WHERE id = %s", (student_id,))
        conn.commit()
        cur.close()
        conn.close()

    return jsonify({"message": "Attendance marked successfully"})

if __name__ == '__main__':
    app.run(debug=True)
# app.py (Backend - Flask)
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import psycopg2
import json

app = Flask(__name__)

# Connect to PostgreSQL
def connect_db():
    conn = psycopg2.connect(
        dbname="attendance_system", 
        user="your_user", 
        password="your_password", 
        host="localhost", 
        port="5432"
    )
    return conn

# Facial Recognition model loading
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('models/facial_recognition_model.h5')

# Student Registration Endpoint
@app.route('/register', methods=['POST'])
def register_student():
    data = request.json  # Get data as JSON
    name = data['name']
    admission_number = data['admission_number']
    email = data['email']
    department = data['department']
    
    # Insert student data into PostgreSQL database
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO students (name, admission_number, email, department) 
        VALUES (%s, %s, %s, %s)
    """, (name, admission_number, email, department))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"message": "Student registered successfully"}), 200

# Facial Recognition and Attendance Marking Endpoint
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.json  # Receive image data as JSON
    image_data = np.array(data['image'], dtype=np.uint8)  # Convert image data to numpy array
    
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = image_data[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))  # Resize for model input
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        prediction = model.predict(face)  # Model inference

        # Mark attendance based on the face recognition result (details depend on your model output)
        student_id = prediction.argmax()  # Assuming the model gives a class index
        
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("UPDATE students SET attendance = TRUE WHERE id = %s", (student_id,))
        conn.commit()
        cur.close()
        conn.close()

    return jsonify({"message": "Attendance marked successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)

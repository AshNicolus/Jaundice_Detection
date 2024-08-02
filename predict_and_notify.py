import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from twilio.rest import Client
from datetime import datetime
import requests
from geopy.geocoders import Nominatim

# Function to get the location of the device
def get_location():
    try:
        response = requests.get('http://ipinfo.io')
        data = response.json()
        coordinates = data['loc']
        return coordinates
    except Exception as e:
        print(f"Error getting location: {e}")
        return "Unknown location"

# Function to get a human-readable address from coordinates
def get_address_from_coordinates(coordinates):
    try:
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.reverse(coordinates, language='en')
        return location.address
    except Exception as e:
        print(f"Error getting address: {e}")
        return "Unknown address"

# Load the trained model
model = load_model('jaundice_detection_model.h5')

# Twilio credentials
account_sid = 'xxxxxxxxxxxxxxxxxx'
auth_token = 'xxxxxxxxxxxxxxxxxx'
client = Client(account_sid, auth_token)

# Initialize the camera (use your device's built-in camera)
camera = cv2.VideoCapture(0)

jaundice_detected = False

# Loop to process frames from the camera feed
while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture image from camera.")
        break

    # Preprocess the frame and make predictions
    processed_frame = cv2.resize(frame, (150, 150))
    processed_frame = img_to_array(processed_frame) / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)
    prediction = model.predict(processed_frame)

    # Show the frame and prediction
    if prediction > 0.5:
        cv2.putText(frame, "Jaundice not Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Jaundice Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        jaundice_detected = True
        break

    # Display the processed frame
    cv2.imshow("Live Camera", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Send Twilio message for jaundice detection after exiting the loop
if jaundice_detected:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    coordinates = get_location()
    address = get_address_from_coordinates(coordinates)
    message = client.messages.create(
        body=f'Jaundice detected at {current_time} on {current_date}. Location: {address}',
        from_='xxxxxxxxx',
        to='xxxxxxxxxxx'
    )

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()

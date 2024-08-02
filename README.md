## Description
This project is a system for detecting jaundice using an image classification model built on top of VGG16. The system captures images from an IP webcam, processes them, and sends an alert using Twilio if jaundice is detected. The model is trained on a dataset containing over 2500 images of jaundice and more than 500 images of normal images to ensure robust image processing and detection.


## Dependencies
This project requires the following Python packages:

Python 3.12.4



- `os` (Standard Library)
- `cv2` (OpenCV) - Version: 4.5.3
- `numpy` - Version: 1.21.0
- `tensorflow` - Version: 2.5.0
- `matplotlib` - Version: 3.4.2
- `twilio` - Version: 6.50.0
- `datetime` (Standard Library)
- `requests` 
- `geopy`
```sh
pip install opencv-python numpy tensorflow matplotlib twilio requests geopy
```

To get the exact versions, you can run the version check code snippets provided for each package in a Python environment.


## Installation

### Clone the Repository
```sh
git clone https://github.com/AshNicolus/Jaundice_Detection.git
cd jaundice_detection
```

Dataset :-
 `"https://drive.google.com/file/d/1-oswkGQI-GCKBFOGZtmVvpmsOTEbHeJE/view?usp=drive_link"` 

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Contact
For questions or collaboration, please contact Ash Nicolus.

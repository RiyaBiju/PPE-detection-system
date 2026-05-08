# PPE Guard

Real-Time Workplace Safety Detection System using YOLOv8 and Flask

---

## Overview

PPE Guard is a computer vision–based safety monitoring system designed to detect whether workers are wearing essential Personal Protective Equipment (PPE) in real time.

The system uses multiple YOLOv8 models for detecting:

- Hard Hats
- Safety Vests
- Gloves
- Safety Shoes
- Goggles

The application supports webcam-based monitoring through a Flask interface and is intended for industrial and construction safety environments.

---

## Features

- Real-time PPE detection
- YOLOv8-based inference
- Flask web interface
- Multi-model PPE monitoring
- OpenCV video processing
- Visual safety status indication

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python |
| Detection Models | YOLOv8 |
| Computer Vision | OpenCV |
| Web Framework | Flask |
| Deep Learning | PyTorch |

---

## Project Structure

```bash
PPE-Guard/
│
├── app_ppe.py
├── detect_webcam.py
├── ppe_regions.py
├── requirements.txt
├── templates/
│
├── models/
│   ├── best.pt
│   ├── vest_best.pt
│   ├── gloves_best.pt
│   ├── shoes_best.pt
│   └── goggles_best.pt
```

---

## Model Weights

Download the trained model weights from the following Google Drive folder:

[Model Weights Drive Folder](https://drive.google.com/drive/folders/1B3tUMfBOo_16n9n49TfH3rpN5-qUhlVM?usp=drive_link&utm_source=chatgpt.com)

Place all `.pt` files inside the project root directory or the `models/` folder.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app_ppe.py
```

Open the application in your browser:

```bash
http://localhost:5000
```

---

## PPE Detection Classes

| PPE Category | Status |
|---|---|
| Hard Hat | Supported |
| Safety Vest | Supported |
| Gloves | Supported |
| Safety Shoes | Supported |
| Goggles | Supported |

---

## Applications

- Construction site monitoring
- Industrial safety compliance
- Automated PPE inspection
- Smart workplace surveillance
- Worker safety analytics

---

## Future Improvements

- Multi-person tracking
- Alert and notification system
- Cloud deployment
- Mobile dashboard integration
- PPE compliance analytics

---

## Author

Developed as a real-time AI-based workplace safety monitoring project using computer vision and deep learning.

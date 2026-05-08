# PPE Guard — Workplace Safety Detection System

Real-time PPE detection using YOLOv8 and Flask.

## Models Required
Download and place in root folder:
- best.pt (Hard Hat)
- vest_best.pt (Safety Vest)
- gloves_best.pt (Gloves)
- shoes_best.pt (Safety Shoes)
- goggles_best.pt (Goggles)
https://drive.google.com/drive/folders/1B3tUMfBOo_16n9n49TfH3rpN5-qUhlVM?usp=drive_link Drive link for model weights.
## Installation
pip install -r requirements.txt

## Run
python app_ppe.py

Open http://localhost:5000

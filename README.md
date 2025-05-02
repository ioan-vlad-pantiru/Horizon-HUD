# 🏍️ Horizon-HUD

Real-time heads-up display (HUD) system for motorcycles using Raspberry Pi 5, OpenCV, and TensorFlow Lite. Highlights traffic signs, pedestrians, and more.

## 🚀 Features

- Object detection using TensorFlow Lite
- Real-time camera input
- HUD display via DLPDLCR230NPEVM projector

## 📦 Setup

```bash
git clone https://github.com/yourusername/Horizon-HUD.git
cd Horizon-HUD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py


## Raspberry setup

```bash
sudo apt update
sudo apt install python3-opencv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
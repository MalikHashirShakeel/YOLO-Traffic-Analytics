import torch
import cv2
import numpy as np
from ultralytics import YOLO

print("PyTorch Version:", torch.__version__)
print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)

print("\nCUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nYOLO Import Successful")
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

def generate_mask(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # print(hsv)
    
    # Define lower and upper bounds for green color in HSV
    lower_color = np.array([0, 0, 255])  # Adjust this threshold as needed
    upper_color = np.array([0, 0, 255]) # Adjust this threshold as needed

    # Create a mask where green areas are white and all other areas are black
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Invert the mask (black to white, white to black)
    mask = cv2.bitwise_not(mask)

    return mask

if __name__ == "__main__":
    
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    test_dir = Path(args.model_paths[0]) / "test"

    # projDir = Path(".")/f"ours_30000" #! need too change
    
    for method in os.listdir(test_dir):
        projDir = test_dir / method
        imageDir = projDir / "gt" 
        saveMaskDir = projDir / "mask" 
        
        saveMaskDir.mkdir(parents=True, exist_ok=True)
        for image_path in imageDir.glob("*.png"):
            mask = generate_mask(str(image_path))
            mask_path = saveMaskDir/f"{image_path.stem}.png"
            cv2.imwrite(str(mask_path), mask)
        
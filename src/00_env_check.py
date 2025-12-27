import platform
import sys

def main():
    print("=== System ===")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Processor:", platform.processor())

    print("\n=== Libraries ===")
    try:
        import cv2
        print("opencv-python:", cv2.__version__)
    except Exception as e:
        print("opencv-python: FAILED", e)

    try:
        import mediapipe as mp
        print("mediapipe:", mp.__version__)
    except Exception as e:
        print("mediapipe: FAILED", e)

    try:
        import ultralytics
        print("ultralytics:", ultralytics.__version__)
    except Exception as e:
        print("ultralytics: FAILED", e)

    print("\n=== PyTorch / Acceleration ===")
    try:
        import torch
        print("torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch: FAILED (YOLO may not run yet)", e)

if __name__ == "__main__":
    main()

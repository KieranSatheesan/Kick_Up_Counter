import time
import cv2
from ultralytics import YOLO

def main():
    # A small model first (faster). You can switch to yolov8m/yolov8l later.
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames += 1
        # Run detection (CPU by default). If you have CUDA it may use it automatically.
        results = model.predict(frame, verbose=False)

        annotated = results[0].plot()

        fps = frames / (time.time() - t0)
        cv2.putText(annotated, f"YOLO OK | FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Check (press q)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

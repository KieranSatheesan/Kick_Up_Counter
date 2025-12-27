import cv2
import time

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try index 1 or remove CAP_DSHOW.")

    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        frames += 1
        dt = time.time() - t0
        fps = frames / dt if dt > 0 else 0

        cv2.putText(frame, f"Webcam OK | FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam Check (press q)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

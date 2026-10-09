from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best.pt"


def main():
    print("=" * 55)
    print("COASTGUARD AI")
    print("Live Plastic Waste Detection")
    print("=" * 55)

    if not MODEL_PATH.exists():
        print("ERROR: Model not found:", MODEL_PATH)
        return

    print("Loading model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully.")

    print("Opening webcam...")
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    print("Webcam opened successfully.")

    frame_count = 0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to capture webcam frame.")
                break

            frame_count += 1
            print("Processing frame:", frame_count)

            results = model.predict(
                source=frame,
                conf=0.50,
                imgsz=640,
                verbose=False,
            )

            annotated_frame = results[0].plot()

            cv2.imshow(
                "CoastGuard AI - Live Detection",
                annotated_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("Stopping live detection.")
                break

    except KeyboardInterrupt:
        print("Detection interrupted.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Total frames processed:", frame_count)
        print("Webcam closed.")


if __name__ == "__main__":
    main()

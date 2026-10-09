from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best.pt"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    print("=" * 60)
    print("COASTGUARD AI")
    print("Video Plastic Waste Detection")
    print("=" * 60)

    video_path = input(
        "\nEnter the full path of the video: "
    ).strip().strip('"')

    video_path = Path(video_path)

    if not video_path.exists():
        print("\nERROR:")
        print(f"Video not found: {video_path}")
        return

    if not MODEL_PATH.exists():
        print("\nERROR:")
        print(f"Model not found: {MODEL_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading CoastGuard AI model...")
    print(f"Model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")
    print("\nRunning video detection...")
    print("Please wait. Video detection can take some time on CPU.")

    results = model.predict(
        source=str(video_path),
        conf=0.50,
        imgsz=640,
        save=True,
        project=str(OUTPUT_DIR),
        name="video_predictions",
        exist_ok=True
    )

    print("\nVideo detection completed.")

    print("\nAnnotated video saved inside:")
    print(OUTPUT_DIR / "video_predictions")

    print("\nResults processed:")
    print(len(results))


if __name__ == "__main__":
    main()
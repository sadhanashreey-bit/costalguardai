from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best.pt"
IMAGE_DIR = Path(r"C:\Users\Admin\Downloads\coastguard_ai_starter\pictures")
OUTPUT_DIR = BASE_DIR / "outputs" / "batch_predictions"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("=" * 60)
    print("COASTGUARD AI")
    print("Batch Plastic Waste Detection")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print("\nERROR: Model not found:")
        print(MODEL_PATH)
        return

    if not IMAGE_DIR.exists():
        print("\nERROR: Image folder not found:")
        print(IMAGE_DIR)
        return

    images = [
        image for image in IMAGE_DIR.iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print("\nNo images found.")
        return

    print(f"\nImages found: {len(images)}")
    print(f"Model: {MODEL_PATH}")
    print(f"Input folder: {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully.")

    print("\nStarting batch detection...\n")

    results = model.predict(
        source=str(IMAGE_DIR),
        conf=0.25,
        imgsz=640,
        save=True,
        project=str(OUTPUT_DIR),
        name="results",
        exist_ok=True
    )

    total_plastic = 0
    total_other = 0
    total_objects = 0

    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)

    for result in results:
        plastic_count = 0
        other_count = 0

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = result.names[class_id]

                if label == "plastic_waste":
                    plastic_count += 1
                    total_plastic += 1

                elif label == "other_waste":
                    other_count += 1
                    total_other += 1

        total_objects += plastic_count + other_count

        print(
            f"{result.path.name}: "
            f"{plastic_count} plastic, "
            f"{other_count} other"
        )

    print("\n" + "-" * 60)
    print(f"Total plastic waste detections: {total_plastic}")
    print(f"Total other waste detections: {total_other}")
    print(f"Total detections: {total_objects}")
    print("-" * 60)

    print("\nAnnotated images saved to:")
    print(OUTPUT_DIR / "results")


if __name__ == "__main__":
    main()
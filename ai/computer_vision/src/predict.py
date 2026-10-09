from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best.pt"
OUTPUT_DIR = BASE_DIR / "outputs"


def detect(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading CoastGuard AI model...")
    print(f"Model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")
    print("\nRunning detection...")

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=640,
        save=True,
        project=str(OUTPUT_DIR),
        name="predictions",
        exist_ok=True
    )

    return results


def print_results(results):
    total_objects = 0

    print("\n" + "=" * 60)
    print("COASTGUARD AI - DETECTION RESULTS")
    print("=" * 60)

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            label = result.names[class_id]

            coordinates = box.xyxy[0].tolist()

            x1, y1, x2, y2 = coordinates

            print(f"\nObject: {label}")
            print(f"Confidence: {confidence:.2f}")
            print(
                f"Bounding Box: "
                f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
            )

            total_objects += 1

    print("\n" + "-" * 60)
    print(f"Total detected objects: {total_objects}")
    print("-" * 60)


def main():

    print("=" * 60)
    print("COASTGUARD AI")
    print("Plastic Waste Detection System")
    print("=" * 60)

    image_path = input(
        "\nEnter the full path of the image: "
    ).strip()

    try:

        results = detect(image_path)

        print_results(results)

        print("\nAnnotated image saved inside:")
        print(OUTPUT_DIR / "predictions")

    except Exception as error:

        print("\nERROR:")
        print(error)


if __name__ == "__main__":
    main()
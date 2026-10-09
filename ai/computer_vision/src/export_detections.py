from pathlib import Path
import csv
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best.pt"
IMAGE_DIR = Path(r"C:\Users\Admin\Downloads\coastguard_ai_starter\pictures")
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"

CSV_FILE = OUTPUT_DIR / "detections.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("=" * 60)
    print("COASTGUARD AI")
    print("Detection CSV Export")
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

    print(f"\nImages found: {len(images)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully.")

    rows = []

    print("\nRunning detection...\n")

    for image in images:
        results = model.predict(
            source=str(image),
            conf=0.50,
            imgsz=640,
            verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = result.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                rows.append({
                    "image": image.name,
                    "class": label,
                    "confidence": round(confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                })

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("CSV EXPORT COMPLETE")
    print("=" * 60)

    print(f"\nTotal detection records: {len(rows)}")

    print("\nCSV saved to:")
    print(CSV_FILE)


if __name__ == "__main__":
    main()
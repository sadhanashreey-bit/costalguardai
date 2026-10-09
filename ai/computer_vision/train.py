from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

DATASET_YAML = BASE_DIR / "data.yaml"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():

    print("=" * 60)
    print("COASTGUARD AI")
    print("YOLO11 Plastic Waste Training")
    print("=" * 60)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found:\n{DATASET_YAML}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading YOLO11 model...")

    model = YOLO("yolo11n.pt")

    print("Model loaded.")
    print("\nStarting training...\n")

    results = model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(OUTPUT_DIR),
        name="coastguard_yolo11",
        exist_ok=True,
        patience=10,
        plots=True
    )

    print("\nTraining completed.")

    print("\nTraining results:")
    print(results)

    trained_model = (
        OUTPUT_DIR
        / "coastguard_yolo11"
        / "weights"
        / "best.pt"
    )

    if trained_model.exists():

        destination = MODEL_DIR / "best.pt"

        destination.write_bytes(
            trained_model.read_bytes()
        )

        print("\nBest model copied to:")
        print(destination)

    else:
        print("\nWARNING:")
        print("best.pt was not found.")


if __name__ == "__main__":
    main()
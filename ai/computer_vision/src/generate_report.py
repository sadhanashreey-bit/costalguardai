from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best.pt"
IMAGE_DIR = Path(r"C:\Users\Admin\Downloads\coastguard_ai_starter\pictures")
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("=" * 60)
    print("COASTGUARD AI")
    print("Detection Statistics Report")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print("ERROR: Model not found.")
        return

    if not IMAGE_DIR.exists():
        print("ERROR: Image folder not found.")
        return

    images = [
        image for image in IMAGE_DIR.iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nImages found: {len(images)}")

    model = YOLO(str(MODEL_PATH))

    total_plastic = 0
    total_other = 0
    highest_plastic = 0
    highest_image = None

    report_lines = []

    for image in images:
        results = model.predict(
            source=str(image),
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        plastic = 0
        other = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = result.names[class_id]

                if label == "plastic_waste":
                    plastic += 1
                elif label == "other_waste":
                    other += 1

        total_plastic += plastic
        total_other += other

        if plastic > highest_plastic:
            highest_plastic = plastic
            highest_image = image.name

        line = f"{image.name}: {plastic} plastic, {other} other"
        report_lines.append(line)

        print(line)

    total_images = len(images)

    if total_images > 0:
        average_plastic = total_plastic / total_images
        average_other = total_other / total_images
    else:
        average_plastic = 0
        average_other = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = OUTPUT_DIR / "detection_report.txt"

    with open(report_file, "w", encoding="utf-8") as file:
        file.write("COASTGUARD AI - DETECTION REPORT\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Images processed: {total_images}\n")
        file.write(f"Total plastic detections: {total_plastic}\n")
        file.write(f"Total other-waste detections: {total_other}\n")
        file.write(
            f"Total detections: "
            f"{total_plastic + total_other}\n"
        )
        file.write(
            f"Average plastic detections per image: "
            f"{average_plastic:.2f}\n"
        )
        file.write(
            f"Average other-waste detections per image: "
            f"{average_other:.2f}\n"
        )

        if highest_image:
            file.write(
                f"Image with most plastic detections: "
                f"{highest_image}\n"
            )
            file.write(
                f"Highest plastic detection count: "
                f"{highest_plastic}\n"
            )

        file.write("\nIMAGE DETAILS\n")
        file.write("-" * 50 + "\n")

        for line in report_lines:
            file.write(line + "\n")

    print("\n" + "=" * 60)
    print("REPORT COMPLETE")
    print("=" * 60)

    print(f"\nTotal plastic detections: {total_plastic}")
    print(f"Total other detections: {total_other}")
    print(f"Total detections: {total_plastic + total_other}")
    print(f"Average plastic per image: {average_plastic:.2f}")

    if highest_image:
        print(f"Most plastic detected in: {highest_image}")
        print(f"Plastic detections: {highest_plastic}")

    print("\nReport saved to:")
    print(report_file)


if __name__ == "__main__":
    main()
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "final_evaluation_report.txt"


def main():
    report = f"""
============

```
         COASTGUARD AI
   FINAL COMPUTER VISION EVALUATION
```

============================================================

1. PROJECT OVERVIEW

---

Project: CoastGuard AI
Purpose: Detect plastic waste and other waste using computer
vision to support coastal pollution monitoring.

Model: Trained YOLO object detection model
Model file: models/best.pt
Classes:

* plastic_waste
* other_waste

2. MODEL VALIDATION METRICS

---

Overall precision: 0.450 (45.0%)
Overall recall: 0.366 (36.6%)
Overall mAP@50: 0.334 (33.4%)
Overall mAP@50-95: 0.227 (22.7%)

Plastic waste:

* Precision: 0.483 (48.3%)
* Recall: 0.438 (43.8%)
* mAP@50: 0.391 (39.1%)
* mAP@50-95: 0.272 (27.2%)

Other waste:

* Precision: 0.418 (41.8%)
* Recall: 0.293 (29.3%)
* mAP@50: 0.277 (27.7%)
* mAP@50-95: 0.182 (18.2%)

Note: These are validation metrics, not overall accuracy.
Results indicate that further model improvement is needed.

3. IMAGE DETECTION TEST

---

Test images processed: 6
Plastic waste detections: 23
Other waste detections: 15
Total detections: 38

Average plastic detections per image: 3.83

Most plastic detections were recorded in:
john-cameron-FMrZLPdDyx4-unsplash.jpg

These are detections at the previously tested confidence
threshold of 0.25. Counts may change at other thresholds.

4. BATCH PROCESSING

---

Status: Successfully tested
Input: Local test image folder
Output: Annotated images and detection results

5. CSV EXPORT

---

Status: Successfully tested
Output file: outputs/reports/detections.csv

The CSV contains detection information such as image name,
class, confidence score, and bounding-box coordinates.

The number of exported records depends on the confidence
threshold used for that export.

6. VIDEO DETECTION

---

Status: Successfully tested
Input: Test video
Output folder: outputs/video_predictions

The model processed video frames and saved detection results.

7. LIVE WEBCAM DETECTION

---

Status: Successfully tested
Webcam opened successfully.
Live frames were captured and processed.
Bounding boxes were displayed during testing.
The test processed 96 frames before being stopped.

8. LIMITATIONS

---

* Model precision and recall need improvement.
* Small, distant, occluded, or visually ambiguous objects
  may be missed or incorrectly classified.
* Performance may vary with lighting, camera quality,
  background, and object size.
* CPU inference may be slow for real-time use.
* Test-image detection counts are not accuracy measurements.
* Field testing is needed before real-world deployment.

9. FUTURE IMPROVEMENTS

---

* Add more diverse coastal waste training images.
* Review and improve dataset labels.
* Train and compare improved model configurations.
* Evaluate precision, recall, and mAP after each change.
* Test across different lighting and camera conditions.
* Integrate detection with the CoastGuard AI backend.
* Add location and timestamp metadata where available.

10. CONCLUSION

---

The computer-vision module has been tested with images,
batch processing, video, CSV export, and live webcam input.

Further model evaluation and backend integration are needed
before considering the system ready for field deployment.

============================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
===================================================================

"""

    REPORT_PATH.write_text(report.strip() + "\n", encoding="utf-8")

    print("=" * 55)
    print("COASTGUARD AI - FINAL EVALUATION")
    print("=" * 55)
    print("Evaluation report created successfully!")
    print("Report location:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()


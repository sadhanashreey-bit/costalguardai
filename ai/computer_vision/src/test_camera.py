import cv2
import time

print("Testing webcam...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
 print("DirectShow failed. Trying default camera mode...")
camera.release()
camera = cv2.VideoCapture(0)

if not camera.isOpened():
 print("ERROR: Webcam could not be opened.")
else:
 print("Webcam opened. Testing frame capture...")


for i in range(10):
    success, frame = camera.read()

    print("Frame", i + 1, "success:", success)

    if success:
        cv2.imshow("Webcam Test - Press Q to close", frame)

        if cv2.waitKey(500) & 0xFF == ord("q"):
            break
    else:
        time.sleep(0.2)

camera.release()
cv2.destroyAllWindows()


print("Webcam test finished.")

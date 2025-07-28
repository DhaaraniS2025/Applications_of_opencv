import cv2
import numpy as np
import requests

url = "http://192.168.1.10:8080/shot.jpg"
cm_per_pixel = 0.04

while True:
    try:
        image_get = requests.get(url, timeout=2)
        arr = np.array(bytearray(image_get.content), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    except requests.exceptions.RequestException:
        print("Unable to fetch frame from IP camera.")
        continue

    frame = cv2.resize(img, (500, 300))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_dark_pink = np.array([140, 50, 100])
    upper_dark_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_dark_pink, upper_dark_pink)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            width_cm = w * cm_per_pixel
            height_cm = h * cm_per_pixel
            area_cm2 = area * (cm_per_pixel ** 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{width_cm:.1f}x{height_cm:.1f} cm | Area: {area_cm2:.1f} cm2",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

            print(f"Object at ({x},{y}) → {width_cm:.2f}cm x {height_cm:.2f}cm | Area: {area_cm2:.2f} cm²")

    cv2.imshow("Measured Frame", frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


import cv2
import numpy as np

# === Load Image ===
image_path = r"C:\Users\Dhaarani S\Pictures\c120.png"  # 🔺 Replace with your image path
image = cv2.imread(image_path)
if image is None:
    print("Image not found. Check the path.")
    exit()

# Resize for consistent processing (optional)
image = cv2.resize(image, (600, 400))

# === Preprocess Image ===
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# === Thresholding (Otsu's Method) ===
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# === Find Contours (Cells) ===
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Count and Classify Cells by Area ===
rbc_count = 0
wbc_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    if area < 100:  # Ignore very small noise
        continue
    elif 100 <= area <= 800:  # RBCs (adjust based on image resolution)
        rbc_count += 1
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 1)  # Green for RBC
    elif area > 800:  # WBCs (larger)
        wbc_count += 1
        cv2.drawContours(image, [cnt], -1, (255, 0, 0), 2)  # Blue for WBC

# === Display Results ===
cv2.putText(image, f"RBC Count: {rbc_count}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(image, f"WBC Count: {wbc_count}", (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imshow("Thresholded Image", cleaned)
cv2.imshow("Blood Cell Counter", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

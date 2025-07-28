import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Dhaarani S\Pictures\WhatsApp Image 2025-06-22 at 11.05.46_f6204a4b.jpg")
cv2.imshow("Image", img)

print("Pixel (100,50) - BGR:", img[50, 100])
print("Average BGR:", np.mean(img, axis=(0, 1)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
print("Avg Gray Intensity:", np.mean(gray))

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist), plt.title("Gray Histogram")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

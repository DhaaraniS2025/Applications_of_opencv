import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Dhaarani S\Pictures\Report pics\IMG-20250714-WA0010.jpg")
img = cv2.resize(img, (800, 600))
clone = img.copy()
drawing = False
shape = 'rectangle'  
ix, iy = -1, -1
def mouse_draw(event, x, y, flags, param):
    global drawing, ix, iy, img, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing, ix, iy = True, x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img = clone.copy()
        if shape == 'rectangle':
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        else:
            radius = int(((x - ix)**2 + (y - iy)**2)**0.5)
            cv2.circle(img, (ix, iy), radius, (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        blurred = cv2.GaussianBlur(clone, (35, 35), 0)
        mask = np.zeros(img.shape[:2], np.uint8)
        if shape == 'rectangle':
            cv2.rectangle(mask, (ix, iy), (x, y), 255, -1)
        else:
            radius = int(((x - ix)**2 + (y - iy)**2)**0.5)
            cv2.circle(mask, (ix, iy), radius, 255, -1)
        inv_mask = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(blurred, blurred, mask=inv_mask)
        focus = cv2.bitwise_and(clone, clone, mask=mask)
        img[:] = cv2.add(result, focus)
        clone[:] = img.copy()
cv2.namedWindow("Draw Blur")
cv2.setMouseCallback("Draw Blur", mouse_draw)
while True:
    cv2.imshow("Draw Blur", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'): shape = 'rectangle'
    elif key == ord('c'): shape = 'circle'
    elif key == ord('z'): img = clone.copy()
    elif key == ord('q'): break

cv2.destroyAllWindows()

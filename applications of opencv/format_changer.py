import cv2
import os

image_path = input("Enter the full image path (with extension): ").strip()
img = cv2.imread(image_path)

if img is None:
    print("Image not found or path is incorrect.")
else:
    # Ask for new format (e.g., .png, .jpg, .bmp)
    new_format = input("Enter new format (e.g., .png, .jpg, .bmp): ").strip().lower()
    
    if not new_format.startswith("."):
        new_format = "." + new_format  
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    folder = os.path.dirname(image_path)
    output_name = os.path.join(folder, base_name + new_format)
    success = cv2.imwrite(output_name, img)
    
    if success:
        print(f"✅ Image saved as {output_name}")
    else:
        print(" Failed to save image.")

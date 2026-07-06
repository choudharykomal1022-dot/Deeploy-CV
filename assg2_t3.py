import cv2
import numpy as np
import torch

# Load the YOLOv5 model (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can use 'yolov5m' for more accuracy

# Load your image of Poland flag
img_path = "C:/Users\Komal\Desktop\Flag_of_Poland.jpg"  # Change this path to your image location
img = cv2.imread(img_path)

# Perform inference (object detection)
results = model(img)

# Fix the color issue for rendered image
rendered_img = results.render()[0]  # Render the results (adds bounding boxes on the image)
rendered_img_bgr = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)  # Convert YOLO's RGB format to OpenCV's BGR

# Save the rendered image to check detection results
output_img_path = "detected_flags_output.jpg"
cv2.imwrite(output_img_path, rendered_img_bgr)
print(f"Result saved at: {output_img_path}")


# Define the function to check for Poland flag
def is_poland_flag(img):
    """
    Determines if the given image matches the Poland flag pattern:
    - The top half must be predominantly white.
    - The bottom half must be predominantly red.
    """
    # Resize image for consistent detection
    img = cv2.resize(img, (500, 300))  # Normalize dimensions
    height, width, _ = img.shape

    # Divide image into top and bottom halves
    top_half = img[:height // 2, :]
    bottom_half = img[height // 2:, :]

    # Calculate the mean BGR (color) values for each half
    top_mean_color = np.mean(top_half, axis=(0, 1))  # (B, G, R) mean values for top
    bottom_mean_color = np.mean(bottom_half, axis=(0, 1))  # (B, G, R) mean values for bottom

    # Poland flag condition:
    # - Top half should be predominantly white (R, G, B all high values)
    # - Bottom half should be predominantly red (R significantly higher than G and B)
    is_top_white = all(c > 200 for c in top_mean_color)  # High (B, G, R > 200) indicates white
    is_bottom_red = bottom_mean_color[2] > 150 and bottom_mean_color[1] < 100 and bottom_mean_color[
        0] < 100  # Red-dominant

    return is_top_white and is_bottom_red


# Detect Poland flag
detected_poland_flag = False

# Process YOLO detections
for *box, confidence, cls in results.pred[0].tolist():
    if confidence > 0.5:  # Filter weak detections
        label = results.names[int(cls)]  # Label for the detected object
        if label in ["flag", "object"]:  # YOLO doesn't specifically detect flags -- adjust labels accordingly
            # Extract bounding box
            x_center, y_center, width, height = box
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Crop the detected flag region
            cropped_img = img[y1:y2, x1:x2]

            # Check if the cropped region matches Poland's flag
            if is_poland_flag(cropped_img):
                detected_poland_flag = True
                print("Poland flag detected!")
                break

# Fallback: Check the entire image if no flag region was detected
if not detected_poland_flag:
    if is_poland_flag(img):
        detected_poland_flag = True
        print("Poland flag detected using fallback logic!")

# Final output if no detection
if not detected_poland_flag:
    print("The image does not contain a Poland flag.")

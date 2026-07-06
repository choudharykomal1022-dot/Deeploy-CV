import cv2
import urllib.request
import os

# Step 1: Download the model files (deploy.prototxt and mobilenet_iter_73000.caffemodel)

# URLs for the model files
prototxt_url = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt'
caffemodel_url = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel'

# File paths to save the model files
prototxt_file = 'deploy.prototxt'
caffemodel_file = 'mobilenet_iter_73000.caffemodel'

# Download the files if they don't exist
if not os.path.exists(prototxt_file):
    print(f"Downloading {prototxt_file}...")
    urllib.request.urlretrieve(prototxt_url, prototxt_file)

if not os.path.exists(caffemodel_file):
    print(f"Downloading {caffemodel_file}...")
    urllib.request.urlretrieve(caffemodel_url, caffemodel_file)

print("Model files downloaded!")

# Step 2: Load the SSD model
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)

# Step 3: Process the video
input_video_path ="C:/Users\Komal\Desktop\WhatsApp Video 2024-12-20 at 15.48.50_85b87b4d.mp4"  # Replace with your video path
cap = cv2.VideoCapture(input_video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output
output_video_path = 'output_video_ssd.mp4'  # Save the processed video in the same directory
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# List of class labels for COCO dataset (for SSD with MobileNet)
class_names = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor'
]

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read frame.")
        break

    # Prepare the frame for the network
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), crop=False)

    # Feed the image to the network
    net.setInput(blob)
    detections = net.forward()

    # Draw bounding boxes and labels on the frame
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Adjust confidence threshold to detect more objects (default is 0.2, lowered to 0.1 here)
        if confidence > 0.1:
            class_id = int(detections[0, 0, i, 1])
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated frame to the output video
    out.write(frame)

    # Display the frame with detections (optional)
    cv2.imshow("SSD Detection", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as: {output_video_path}")

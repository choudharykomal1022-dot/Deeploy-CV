import torch
import cv2

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Path to your input video
input_video_path = "C:/Users\Komal\Desktop\WhatsApp Video 2024-12-20 at 15.48.50_85b87b4d.mp4"  # Replace with your video path
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
output_video_path = 'output_video.mp4'  # Save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read frame.")
        break

    # Perform object detection
    results = model(frame)

    # Annotate the frame with detections
    annotated_frame = results.render()[0]

    # Save the annotated frame to the output video
    out.write(annotated_frame)

    # Display the frame with detections (optional)
    cv2.imshow("YOLOv5 Detection", annotated_frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as: {'c:/Users/Komal/Desktop'}")

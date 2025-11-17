import cv2  # OpenCV for image processing
import imutils  # Helps resize frames
import numpy as np  # NumPy for numerical calculations
import math  # For distance calculation between people
from itertools import combinations  # To compare all pairs of people

# Load the pre-trained MobileNet SSD model
model = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

# Object classes (we only care about "person")
classesList = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                  "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                  "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Function to detect people and check social distancing
def detect_social_distancing():
    video = cv2.VideoCapture('SampleOne.mp4')  # Load video file

    while True:
        success, frame = video.read()  # Read a frame
        if not success:
            break  # Stop if video ends

        frame = imutils.resize(frame, width=600)  # Resize for faster processing
        height, width = frame.shape[:2]  # get width and height from frame it will ignore colors from frame 
        # (720, 1280, 3)  # height=720, width=1280, 3 color channels (RGB)

        # Convert the frame into a format the model can process
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
        # 0.007843 It is used to normalize pixel values from the range [0, 255] to [−1, 1].
        # If we keep raw pixel values [0, 255], the model may struggle to learn efficiently.
        # 
        model.setInput(blob)  # Set as model input
        detections = model.forward()  # runs the object detection on the frame and stores the results in detections.

        detected_people = {}  # Store detected people with their positions

        for i in range(detections.shape[2]):  # Loop through detected objects
            confidence = detections[0, 0, i, 2]  # Get confidence score

            # detections[0] → Accesses the first batch (since we process one image at a time).
            # detections[0, 0] → Accesses the first set of detections.
            # detections[0, 0, i] → Accesses the i-th detected object.
            # detections[0, 0, i, 2] → Extracts the confidence score for the i-th object.

            if confidence > 0.80:  # Only keep high-confidence detections
                class_index = int(detections[0, 0, i, 1])  # Get object class
                if classesList[class_index] == "person":  # Check if it's a person
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height]) # 3:7 means 3,4,5,6 which 3 is X1 left boundry
                    # 4 is Y1 Top Boundry, 5 is X2 Right and Y2 is Bottom boundry
                    # np.array is scaling the normalized values to pixel values
                    x1, y1, x2, y2 = box.astype("int")  # Convert to integers
                    detected_people[i] = ((x1 + x2) // 2, (y1 + y2) // 2, x1, y1, x2, y2)  # Store person data
                    # x1, y1, x2, y2 represent the bounding box of the detected person.
                    # (x1 + x2) // 2 → Computes the X-coordinate of the center of the bounding box.
                    # (y1 + y2) // 2 → Computes the Y-coordinate of the center of the bounding box.

        people_in_red_zone = set()  # Store IDs of people who are too close

        # Compare distances between all detected people
        for (id1, person1), (id2, person2) in combinations(detected_people.items(), 2): # Creates all possible pairs of detected people.
            distance = math.hypot(person1[0] - person2[0], person1[1] - person2[1])  # Calculate distance
            if distance < 75:  # If too close, mark them
                people_in_red_zone.add(id1)
                people_in_red_zone.add(id2)

        # Draw bounding boxes
        for person_id, person_box in detected_people.items():
            box_color = (0, 0, 255) if person_id in people_in_red_zone else (0, 255, 0)  # Red for violation, green otherwise
            cv2.rectangle(frame, (person_box[2], person_box[3]), (person_box[4], person_box[5]), box_color, 2)

        # Show the video with detections
        cv2.imshow("Social Distancing Detection", frame)

        # Press 'a' to exit
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    video.release()  # Release the video
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Run the function
detect_social_distancing()

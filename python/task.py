import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Paths to the YOLO files
cfg_path = 'yolov4.cfg'
weights_path = 'yolov4.weights'

# Initialize YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Adjust indices based on OpenCV version
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Define database connection
conn = sqlite3.connect('people_count.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS people
             (id INTEGER PRIMARY KEY, timestamp TEXT)''')

def detect_people(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                if obj.size == 85:
                    obj = np.array(obj)
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    confidence = obj[4]
                    scores = obj[5:]
                    
                    if confidence > 0.5:
                        class_id = np.argmax(scores)
                        if class_id == 0:
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indices, boxes

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide IP camera URL

    # Define codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'MJPG', 'XVID', 'MP4V', etc.
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Adjust the resolution and FPS as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        indices, boxes = detect_people(frame)
        if indices is not None and len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute('INSERT INTO people (timestamp) VALUES (?)', (timestamp,))
            conn.commit()
            print(f"Inserted timestamp: {timestamp}")

        # Write the frame to the video file
        out.write(frame)

        # Display the frame
        cv2.imshow('Video Feed', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything if the job is finished
    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import threading
import queue
import time
import torch
from flask import Flask, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Camera configurations
camera_slots = {
     "camera1": {
        "slots": {
            "slot_1": np.array([[78, 202], [330, 213], [364, 355], [6, 353]]),
        },
        "rtsp_url": "rtsp://admin:hik@13579@121.46.84.105:554/Streaming/Channels/502/",
    },
    "camera2": {
        "slots": {
 "slot_2": np.array([[382, 5], [588, 25], [586, 79], [330, 40]]),
    "slot_3": np.array([[308, 36], [241, 83], [572, 143], [581, 84]]),
    "slot_4": np.array([[227, 86], [124, 160], [556, 275], [566, 142]]),
    "slot_5": np.array([[114, 155], [5, 235], [159, 355], [552, 298]])
        },
        "rtsp_url": "rtsp://admin:hik@13579@121.46.84.105:554/Streaming/Channels/302/",
    },
    "camera3": {
        "slots": {
       "slot_6": np.array([[ 639, 91], [  980, 166], [ 980, 209], [569, 136]]),
    "slot_7": np.array([[561, 135], [ 986, 221], [989, 304], [ 495, 173]]),
    "slot_8": np.array([[495, 178], [ 986, 312], [974, 416], [371, 228]]),
    "slot_9": np.array([[366, 235], [ 966, 424], [938, 549], [ 213, 287]]),
    "slot_10": np.array([[212, 299], [934, 560], [ 911, 663], [80, 363]]),
     "slot_11": np.array([[74, 367], [889, 684], [  0, 704], [4, 415]]),
        },
        "rtsp_url": "rtsp://admin:hik@13579@121.46.84.105:554/Streaming/Channels/602/",
    },
    "camera4": {
        "slots": {
    "slot_12": np.array([[221, 29], [488, 78], [505, 19], [280, 4]]),
    "slot_13": np.array([[488, 72], [221, 29], [161, 73], [489, 171]]),
      "slot_14": np.array([[162, 75],  [96, 119],[ 462, 266], [ 490, 178]]),
       "slot_15": np.array([[98, 121],  [463, 267],[404, 351], [  19, 180]]),
     "slot_16": np.array([[ 22, 183],  [4, 192],[9, 347], [382, 347]])
        },
        "rtsp_url": "rtsp://admin:hik@13579@121.46.84.105:554/Streaming/Channels/102/",
    },
    "camera5": {
        "slots": {
            "slot_17": np.array([[342, 48], [378, 48], [429, 100], [355, 113]]),
            "slot_18": np.array([[290, 51], [260, 118], [322, 115], [330, 40]]),
        },
        "rtsp_url": "rtsp://admin:hik@13579@121.46.84.105:554/Streaming/Channels/202/",
    },
    "camera6": {
        "slots": {
        "slot_19": np.array([[442, 127],[551, 205], [133, 327], [103, 227]]),
    "slot_20": np.array([[112, 222], [407, 131], [332, 81], [94, 128]]),
    "slot_21": np.array([[93, 128], [319, 87], [278, 63], [89, 90]])
        },
        "rtsp_url": "rtsp://admin:hik@13579@121.46.84.105:554/Streaming/Channels/402/",
    },
}

# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = torch.hub.load('./yolov5', 'yolov5s', source='local').to(device)


# Global variables
all_slots_status = {}
camera_queues = {}
camera_threads = {}
processing_flags = {}

def is_car_in_slot(car_bbox, slot_points):
    """Check if a car is inside a parking slot."""
    car_x_center = (car_bbox[0] + car_bbox[2]) / 2
    car_y_center = (car_bbox[1] + car_bbox[3]) / 2
    point = (car_x_center, car_y_center)
    result = cv2.pointPolygonTest(slot_points.astype(np.float32), point, False)
    return result >= 0

def detect_cars(frame):
    """
    Detect cars using YOLOv5.
    
    Args:
        frame (numpy.ndarray): The input image/frame.
    
    Returns:
        list: List of bounding boxes for detected cars.
    """
    # Convert frame to RGB (YOLOv5 expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = yolo_model(frame_rgb)
    
    # Extract bounding boxes
    car_boxes = []
    for *box, conf, cls in results.xyxy[0]:  # bbox format: [x1, y1, x2, y2]
        class_id = int(cls)
        if results.names[class_id] in ["car", "motorbike"]:  # Check if it's a car or motorbike
            x1, y1, x2, y2 = map(int, box)
            car_boxes.append([x1, y1, x2, y2])
    
    return car_boxes

def process_camera_feed(camera_name, camera_config):
    global all_slots_status, processing_flags
    rtsp_url = camera_config["rtsp_url"]
    slots = camera_config["slots"]
    frame_queue = camera_queues[camera_name]
    
    while True:
        try:
            if processing_flags.get(camera_name, False):
                time.sleep(1)
                continue

            processing_flags[camera_name] = True

            if frame_queue.empty():
                processing_flags[camera_name] = False
                time.sleep(1)
                continue

            frame = frame_queue.get()
            
            car_boxes = detect_cars(frame)
            slot_status = {}

            for slot_name, slot_points in slots.items():
                slot_occupied = False
                for car_box in car_boxes:
                    if is_car_in_slot(car_box, slot_points):
                        slot_occupied = True
                        break
                
                slot_status[slot_name] = "Occupied" if slot_occupied else "Free"

            all_slots_status[camera_name] = slot_status
            print(f"Updated slot status for {camera_name}: {slot_status}")

            processing_flags[camera_name] = False
            time.sleep(5)

        except Exception as e:
            print(f"Error processing {camera_name}: {e}")
            processing_flags[camera_name] = False
            time.sleep(10)

def capture_camera_feed(camera_name, camera_config):
    rtsp_url = camera_config["rtsp_url"]
    frame_queue = camera_queues[camera_name]
    
    while True:
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"Failed to open stream for {camera_name}")
                time.sleep(10)
                continue

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                while not frame_queue.empty():
                    frame_queue.get()
                
                frame_queue.put(frame)
                time.sleep(0.1)

            cap.release()
            time.sleep(10)

        except Exception as e:
            print(f"Error capturing feed for {camera_name}: {e}")
            time.sleep(10)

@app.route('/', methods=['GET'])
def get_all_slot_status():
    return jsonify(all_slots_status)

if __name__ == "__main__":
    for camera_name in camera_slots:
        camera_queues[camera_name] = queue.Queue(maxsize=1)
        processing_flags[camera_name] = False

    for camera_name, camera_config in camera_slots.items():
        capture_thread = threading.Thread(
            target=capture_camera_feed, 
            args=(camera_name, camera_config), 
            daemon=True
        )
        capture_thread.start()
        camera_threads[camera_name + '_capture'] = capture_thread

    for camera_name, camera_config in camera_slots.items():
        process_thread = threading.Thread(
            target=process_camera_feed, 
            args=(camera_name, camera_config), 
            daemon=True
        )
        process_thread.start()
        camera_threads[camera_name + '_process'] = process_thread

    app.run(host='0.0.0.0', port=5000)

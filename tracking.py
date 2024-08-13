import cv2
import math
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Obj:
    id: int
    type: str
    prev_pos: Tuple[int, int]

model = YOLO("model/yolov10n.pt")
video = cv2.VideoCapture("videos/traffic.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
size = (1920//2, 1080//2)
output = cv2.VideoWriter("videos/output.mp4", fourcc, 20, size)
count = 1
prev_center_points = []
prev_obj = []
COLORS = [
   (0, 'person', (189, 224, 255)), # Light skin tone, a neutral human representation
   (1, 'bicycle', (200, 0, 0)), # Blue for bicycles, often seen blue in many logos and brands
   (2, 'car', (0, 200, 0)), # Green, representing eco-friendly transportation
   (3, 'motorcycle', (255, 140, 0)), # Dark orange, a common color for motorcycles
   (4, 'airplane', (128, 128, 128)), # Grey, resembling the material of many airplanes
   (5, 'bus', (255, 255, 0)), # Yellow, typical for school buses
   (6, 'train', (139, 69, 19)), # Brown, associated with older trains and railways
   (7, 'truck', (0, 128, 0)), # Dark green, frequently seen in logistics and delivery trucks
   (8, 'boat', (0, 255, 255)), # Cyan, representing water associations
   (9, 'traffic light', (173, 255, 47)), # Bright green, similar to the green light
   (10, 'fire hydrant', (0, 0, 255)), # Red because fire hydrants are usually red
   (11, 'stop sign', (0, 0, 139)), # Dark red, universally representing "Stop"
   (12, 'parking meter', (192, 192, 192)), # Silver, representing metal parking meters
   (13, 'bench', (160, 82, 45)), # Saddle brown, resembling wooden benches
   (14, 'bird', (255, 215, 0)), # Gold, common in many colorful birds
   (15, 'cat', (128, 0, 128)), # Purple, since cats come in varied colors and purple stands out
   (16, 'dog', (165, 42, 42)), # Brown, typical for many common breeds
   (17, 'horse', (105, 105, 105)), # Dim grey, representing the natural coat of many horses
   (18, 'sheep', (255, 255, 240)), # Ivory, similar to sheep's wool
   (19, 'cow', (255, 250, 250)), # Snow, referring to the white spots on Holstein cows
   (20, 'elephant', (169, 169, 169)), # Dark grey, typical color of elephants
   (21, 'bear', (139, 69, 19)), # Saddle brown, resembling bears' fur
   (22, 'zebra', (0, 0, 0)), # Black, representing zebras' stripes
   (23, 'giraffe', (255, 235, 205)), # Blanched almond, representing the giraffe's lighter patches
   (24, 'backpack', (255, 69, 0)), # Orange red, common backpack color for visibility
   (25, 'umbrella', (138, 43, 226)), # Blue violet, representing a popular umbrella shade
   (26, 'handbag', (219, 112, 147)), # Pale violet red, common handbag color
   (27, 'tie', (0, 0, 128)), # Navy, a typical tie color
   (28, 'suitcase', (210, 105, 30)), # Chocolate, reflecting leather suitcases
   (29, 'frisbee', (255, 165, 0)), # Orange, commonly seen for frisbees
   (30, 'skis', (75, 0, 130)), # Indigo, common for skiing gear
   (31, 'snowboard', (123, 104, 238)), # Medium slate blue, popular color for snowboard designs
   (32, 'sports ball', (255, 105, 180)), # Hot pink, a noticeable color for various sports balls
   (33, 'kite', (72, 209, 204)), # Medium turquoise, reflecting common vibrant kite colors
   (34, 'baseball bat', (245, 245, 220)), # Beige, typical for wooden baseball bats
   (35, 'baseball glove', (139, 69, 19)), # Saddle brown, typical baseball glove color
   (36, 'skateboard', (0, 250, 154)), # Medium spring green, representing vibrant skateboard decks
   (37, 'surfboard', (255, 20, 147)), # Deep pink, common for surfboards
   (38, 'tennis racket', (47, 79, 79)), # Dark slate grey, typical racket color
   (39, 'bottle', (0, 100, 0)), # Dark green, common for glass bottles
   (40, 'wine glass', (255, 99, 71)), # Tomato, often representing the red wine
   (41, 'cup', (220, 20, 60)), # Crimson, a stand-out color for cups
   (42, 'fork', (192, 192, 192)), # Silver, typical for metal cutlery
   (43, 'knife', (211, 211, 211)), # Light grey, typical for steel knives
   (44, 'spoon', (192, 192, 192)), # Silver, reflecting the common metal
   (45, 'bowl', (255, 228, 196)), # Bisque, a light bowl color
   (46, 'banana', (255, 255, 0)), # Yellow, like the fruit
   (47, 'apple', (255, 0, 0)), # Red, like a Red Delicious apple
   (48, 'sandwich', (210, 180, 140)), # Tan, matching the bread
   (49, 'orange', (255, 165, 0)), # Orange, like the fruit
   (50, 'broccoli', (0, 255, 0)), # Bright green, like the vegetable
   (51, 'carrot', (255, 140, 0)), # Dark orange, like the vegetable
   (52, 'hot dog', (184, 134, 11)), # Dark goldenrod, representing the bun
   (53, 'pizza', (255, 223, 0)), # Golden yellow, representing cheese
   (54, 'donut', (205, 92, 92)), # Indian red, representing glazed donuts
   (55, 'cake', (222, 184, 135)), # Burly wood, representing the cake base
   (56, 'chair', (210, 105, 30)), # Chocolate, common wood color
   (57, 'couch', (139, 69, 19)), # Saddle brown, representing leather couches
   (58, 'potted plant', (34, 139, 34)), # Forest green, representing the leaves
   (59, 'bed', (139, 0, 0)), # Dark red, commonly heavy wooden beds
   (60, 'dining table', (160, 82, 45)), # Sienna, typical wood table color
   (61, 'toilet', (255, 245, 238)), # Seashell, representing a clean toilet
   (62, 'tv', (0, 0, 0)), # Black, common for TV screens
   (63, 'laptop', (105, 105, 105)), # Dim grey, representing modern devices
   (64, 'mouse', (169, 169, 169)), # Dark grey, typical for computer accessories
   (65, 'remote', (25, 25, 112)), # Midnight blue, common remote colors
   (66, 'keyboard', (128, 128, 128)), # Grey, typical for keyboards
   (67, 'cell phone', (0, 0, 128)), # Navy, representing tech gadgets
   (68, 'microwave', (128, 128, 128)), # Grey, common appliance color
   (69, 'oven', (169, 169, 169)), # Dark grey, reflecting stainless steel
   (70, 'toaster', (119, 136, 153)), # Light slate grey, typical for toasters
   (71, 'sink', (192, 192, 192)), # Silver, typical for sinks
   (72, 'refrigerator', (255, 255, 255)), # White, common for refrigerators
   (73, 'book', (139, 0, 0)), # Dark red, commonly bound book
   (74, 'clock', (255, 255, 0)), # Yellow, resembling alarm clocks
   (75, 'vase', (255, 182, 193)), # Light pink, representing flower vases
   (76, 'scissors', (165, 42, 42)), # Brown, representing plastic gripped scissors
   (77, 'teddy bear', (160, 82, 45)), # Sienna, common teddy bear color
   (78, 'hair drier', (255, 99, 71)), # Tomato, vibrant color for the device
   (79, 'toothbrush', (0, 191, 255)), # Deep sky blue, commonly associated color
]


def addObj(obj_type, pos):
    global count
    min = 20
    for obj in prev_obj:
        distance = math.hypot(obj.prev_pos[0] - pos[0], obj.prev_pos[1] - pos[1])
        if obj.type == obj_type and distance <= min:
            obj.prev_pos = pos
            return obj.id
    new_obj = Obj(count, obj_type, pos)
    prev_obj.append(new_obj)
    count += 1
    return new_obj.id

success, frame = video.read()
while success:
    h, w, layers = frame.shape
    new_h = h//2
    new_w = w//2
    frame = cv2.resize(frame, (new_w, new_h))

    cur_center_points = []
    results = model.predict(frame, stream=True)
    for res in results:
        classes = res.names
        for box in res.boxes:
            if box.conf[0] > 0.5:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx = (x1 + x2)/2
                cy = (y1 + y2)/2
                cur_center_points.append((cx, cy))
                class_id = int(box.cls[0])
                obj_name = classes[class_id]
                id = addObj(obj_name, (cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[class_id][2], 2)                   
                cv2.putText(frame, str(id), (x1+3, y1+15),
                            fontFace=cv2.FONT_HERSHEY_PLAIN, 
                            fontScale=1,
                            color=(0, 0, 200), 
                            thickness=2)
    output.write(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27: break
    success, frame = video.read()

video.release()
output.release()
cv2.destroyAllWindows()


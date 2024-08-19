# Object-Detection-and-Tracking
### About
This project is an object detection and tracking application built using Python, OpenCV, and YOLOv10. The app is designed to detect various objects from the COCO dataset, including people, vehicles, and other commonly encountered items in video footage.

### Features
- Multi-Object Detection: The app is capable of detecting and distinguishing between multiple types of objects in real-time, such as people, bicycles, cars, and more.
- Object Tracking: Each detected object is tracked throughout the video, with a bounding box color-coded according to the type of object (e.g., light skin tone for people, blue for bicycles, green for cars).
- Object Counting: The app counts the number of detected objects and displays this number alongside each identified object.
- Customizable Bounding Boxes: Different colors are used for bounding boxes to visually distinguish between different types of objects.
How It Works
The app processes video input and uses YOLOv10 for object detection. Each frame is analyzed, and the detected objects are classified and tracked throughout the video. The number of occurrences of each object type is counted and displayed within the bounding box.

### Demo
Check out a demo video of the app in action below:
![](videos/output.mp4)

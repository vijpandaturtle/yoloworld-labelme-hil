import argparse
import json
import os

import numpy as np
from ultralytics import YOLOWorld


class YOLOWorld2LabelMe(object):
    def __init__(self, model, folder_path, image_width, image_height, classes):
        self.folder_path = folder_path
        self.image_width = image_width
        self.image_height = image_height
        self.classes = classes
        self.mapping = {}

        # Initializing classes
        self.model = YOLOWorld(model)
        self.model.set_classes(classes)
        self.create_id2label()
    
    def create_id2label(self):
        self.mapping = {i: label for i, label in enumerate(self.classes)}
    
    def convert_id2label(self, cls):
        return self.mapping[cls]
    
    def model_predict(self, image_path):
        results = self.model.predict(image_path)
        return results

    def convert_yolo_to_labelme(self, results, image_name, image_path):
        custom_format = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],
            "imagePath": f"{image_name}.jpg",
            "imageData": None,
            "imageHeight": self.image_height,
            "imageWidth": self.image_width
        }

        yolo_boxes = results[0]
        print(self.mapping)
        for i in range(len(yolo_boxes.boxes.cls)):
            label = self.convert_id2label(int(yolo_boxes.boxes.cls[i].detach().numpy()))
            x_min, y_min, x_max, y_max = np.float64(yolo_boxes.boxes.xyxy[i].detach().numpy())
            points = [
                [x_min, y_min],
                [x_max, y_max]
            ]
            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }

        custom_format["shapes"].append(shape)
        return custom_format

    def process_folder(self):
        # Iterate over all files in the folder
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    image_name, _ = os.path.splitext(file)
                    image_path_output = image_path.strip('.jpg')

                    # Perform inference and convert to custom format
                    results = self.model_predict(image_path)
                    labelme_format = self.convert_yolo_to_labelme(results, image_name, image_path)

                    json_filename = f"{image_path_output}.json"
                    with open(json_filename, 'w') as json_file:
                        json.dump(labelme_format, json_file, indent=2)
    
                    print(f"Processed: {image_path}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Convert images annotated by YOLO-World bounding boxes to labelme images for assisted labelling.")
    parser.add_argument("--model", required=False, help="Name of the YOLO model weights file", default="yolov8x-world.pt")
    parser.add_argument("--folder", required=True, help="Path to the folder containing images.")
    parser.add_argument("--width", type=int, required=True, help="Image width.")
    parser.add_argument("--height", type=int, required=True, help="Image height.")
    args = parser.parse_args()
  
    classes = ["person", "rider", "animal", "bus", "motorcycle", "truck", "car", "ambulance", "rickshaw", "traffic sign", "traffic light"]
    
    labeller = YOLOWorld2LabelMe(args.model, args.folder, args.width, args.height, classes)
    labeller.process_folder()

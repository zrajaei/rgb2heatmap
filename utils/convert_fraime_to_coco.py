import json
import cv2
from typing import Dict
import numpy as np

def read_json(path:str)->Dict:
    """
    Reads and parses a JSON file from the specified file path.

    Args:
        path (str): The file path to the JSON file to be read.

    Returns:
        Dict: A dictionary containing the data parsed from the JSON file.
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def write_json(data:Dict, path:str)->None:
    """
    Writes a Python dictionary to a JSON file at the specified file path.

    Args:
        data (Dict): The dictionary data to be written to the JSON file.
        path (str): The file path where the JSON file will be saved.

    Returns:
        None
    """
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def convert_fraime_to_coco(input_annotation_path:str, output_annotation_path:str, image_path:str)->None:
    

    def is_bbox(points:Dict)->bool:
        
        if len(points['x'])==2:
            return True
        
        return False
   
    input_annotation_path=input_annotation_path
    image_path=image_path
    output_annotation_path =output_annotation_path

    # Initialize COCO json    
    coco_data = {
        "images":[],
        "annotations":[],
        "categories":[]
    }


    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    annotation_data = read_json(input_annotation_path)

    image_entity = {
        "id":0,
        "file_name":annotation_data['fileName'],
        "height":image_height,
        "width":image_width
    }

    coco_data['images'].append(image_entity)

    annotation_id = 0
    for region in annotation_data['regions']:

        if is_bbox(region['points']):    
            x_min, x_max = region['points']['x'][0], region['points']['x'][1]
            y_min, y_max = region['points']['y'][0], region['points']['y'][1]
            
            # Note: bbox = [x_min,, y_min, width, height]
            bbox = [x_min,y_min, x_max-x_min,y_max-y_min]
            segmentation = []
        else:
            xs = region['points']['x']
            x_min, x_max = np.min(xs), np.max(xs)
            ys = region['points']['y']
            y_min, y_max = np.min(ys), np.max(ys)

            # Note: bbox = [x_min,, y_min, width, height]
            bbox = [x_min,y_min, x_max-x_min,y_max-y_min]
            
            polygons = [value for point in zip(xs, ys) for value in point]
            segmentation=[polygons]
        
        annotation_entity={
            "id": annotation_id,
            "image_id":0,
            "category_id":region['class'],
            "bbox":bbox,
            "area":bbox[2]*bbox[3],
            "segmentation":segmentation,
            "iscrowd":0
        }

        coco_data['annotations'].append(annotation_entity)
        annotation_id+=1

    labels = annotation_data['classNames']

    for id, name in labels.items():
        category_entity={
            "id":id,
            "name":name
        }
        coco_data['categories'].append(category_entity)

    write_json(coco_data, output_annotation_path)
    
def main():
    
    input_annotation_path="/path/to/input-annotation.json"
    output_annotation_path = "/path/to/output-annotation.json"
    image_path="/path/to/image"
   
    convert_fraime_to_coco(input_annotation_path,
                           output_annotation_path,
                           image_path)

if __name__=="__main__":
    main()
import numpy as np
import cv2
from typing import Tuple
import time

class Heatmap():
    
    def __init__(self):
        pass


    def run(self, image: np.ndarray, bboxes: np.ndarray, area_ratio:float=0.1, sigma_ratio :float=0.333) -> np.ndarray:
        """
        Generates and overlays a Gaussian-based heatmap on the input image for each bounding box.

        For each bounding box:
            1. A central region is extracted based on the `area_ratio`.
            2. A random pixel is selected from this central region to serve as the Gaussian center.
            3. A 2D Gaussian heatmap is generated with spread determined by `sigma_ratio` and 
            the bounding box size.
            4. The heatmap is accumulated into the final result using a pixel-wise maximum to 
            handle overlapping bounding boxes.
            5. The final heatmap is normalized and visualized as an overlay on the input image.

        Args:
            image (np.ndarray): Input image on which the heatmap will be overlaid.
            bboxes (np.ndarray): Array of bounding boxes, each defined as (x, y, width, height).
            area_ratio (float, optional): Fraction of the bounding box area used to define 
                the central region for random point selection. Default is 0.1 (i.e., 10% area).
            sigma_ratio (float, optional): Ratio used to compute the standard deviation 
                of the Gaussian relative to the bounding box size. Default is 0.333.

        Returns:
            np.ndarray: The input image with the final heatmap drawn on top as an overlay.
        """

        image_height, image_width = image.shape[:2]
        heatmap = np.zeros((image_height,image_width), dtype=np.float32)

        xs = np.arange(0, image_width, 1)
        ys = np.arange(0, image_height, 1)
        grid_x, grid_y = np.meshgrid(xs,ys)
                
        for bbox in bboxes:
            
            seed = int (time.time())
            np.random.seed(seed)
            
            # Crop the central region from the bbox
            central_region = self.__extract_central_region(bbox, area_ratio)
            
            # Select a random point as center of the bbox to generate its heatmap
            selected_point = self.__select_random_point(central_region)
            
            # Generate the bbox's heatmap
            temporary_heatmap = self.__generate_heatmap(bbox,
                                              selected_point,
                                              grid_x,
                                              grid_y,
                                              sigma_ratio)
            # Note : If two or more bounding boxes overlap, the maximum value among them is used as the heatmap value in the image
            heatmap = np.maximum(heatmap, temporary_heatmap)
        
        # Normalize the heatmap values
        normalized_heatmap = self.__normalize(heatmap)

        # Draw the heatmap on the image
        overlayed_heatmap = self.draw_heatmaps(image,
                                              normalized_heatmap)
        
        return overlayed_heatmap

    def __normalize(self, heatmap:np.ndarray)->np.ndarray:
        """
        Applies min-max normalization to the input heatmap, scaling values 
        to the range [0, 1]. If all values in the heatmap are equal, the 
        original heatmap is returned unchanged to avoid division by zero.

        Args:
            heatmap (np.ndarray): Input heatmap array of arbitrary shape 
                with numerical values.

        Returns:
            np.ndarray: Normalized heatmap with values scaled between 0 and 1,
                or the original heatmap if it has a constant value.
        """
        max_value = np.max(heatmap)
        min_value = np.min(heatmap)

        if max_value>min_value:
            normalized_heatmap = (heatmap-min_value)/(max_value-min_value)
            return normalized_heatmap

        return heatmap
    
    def __generate_heatmap(self, bbox:np.ndarray, point:Tuple[int,int], grid_x: np.ndarray, grid_y:np.ndarray, sigma_ratio:float)->np.ndarray:
        """
        Generates a 2D Gaussian heatmap centered around a given point within a bounding box.

        The Gaussian distribution is shaped according to the bounding box dimensions and 
        scaled by a randomly perturbed `sigma_ratio`. This allows the heatmap to adapt 
        to the size of the bounding box while introducing slight randomness to the spread.

        Args:
            bbox (np.ndarray): Bounding box defined as (x, y, width, height), where
                (x, y) is the top-left corner and (width, height) are the dimensions.
            point (Tuple[int, int]): Center point (x, y) of the Gaussian within the bounding box.
            grid_x (np.ndarray): X-coordinate meshgrid of the image (same shape as output heatmap).
            grid_y (np.ndarray): Y-coordinate meshgrid of the image (same shape as output heatmap).
            sigma_ratio (float): Ratio used to scale the standard deviation of the Gaussian
                relative to the width and height of the bounding box.

        Returns:
            np.ndarray: 2D array representing the generated heatmap, with values in [0, 1].
        """
        
        bbox_width, bbox_height = bbox[2], bbox[3]

        
        alpha = np.random.uniform((sigma_ratio-0.1), (sigma_ratio+0.1))
        sigma_x = alpha * bbox_width
        sigma_y = alpha * bbox_height


        heatmap = np.exp(-(((grid_x - point[0])**2) / (2 * sigma_x**2) + 
                           ((grid_y - point[1])**2) / (2 * sigma_y**2)))
        
        return heatmap


    def __select_random_point(self, bbox:np.ndarray)-> Tuple[int, int]:
        """
        Selects a random pixel inside the bounding box uniformly by sampling 
        a random integer index in the area, then converting to 2D coordinates.

        Args:
            bbox (np.ndarray): Bounding box as (x, y, width, height),
                where (x, y) is the top-left corner.

        Returns:
            tuple[int, int]: Integer coordinates (x_rand, y_rand) of the randomly
                selected pixel inside the bounding box.
        """
        bbox_x, bbox_y, bbox_width, bbox_height = bbox

        bbox_area = bbox_width * bbox_height

        selected_pixel = np.random.randint(0,bbox_area)

        offset_x = selected_pixel % bbox_width
        offset_y = selected_pixel // bbox_width

        selected_point = (bbox_x+offset_x, bbox_y+offset_y)

        return selected_point

    def __extract_central_region(self, bbox: np.ndarray, area_ratio: float)-> np.ndarray:
        """
        Crops a square region from the center of a bounding box based on a specified area ratio.

        This function calculates a smaller square region whose area is a fraction of the original 
        bounding box's area. The resulting region is centered within the original bounding box, 
        and both its width and height are equal. The coordinates of the returned region are 
        relative to the top-left corner of the original bounding box.

        Args:
            bbox (np.ndarray): A NumPy array representing the bounding box in 
                (x, y, width, height) format, where (x, y) is the top-left corner.
            area_ratio (float): A value between 0 and 1 indicating the ratio of the 
                cropped region's area to the original bounding box area.

        Returns:
            np.ndarray: A NumPy array in (x, y, width, height) format representing the 
                cropped square region, with integer values and centered within the original bounding box.
        """

        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        bbox_area= bbox_width * bbox_height

        central_region_area = bbox_area * area_ratio
        central_region_side = np.sqrt(central_region_area)

        center_x = bbox_x+(bbox_width/2)
        center_y = bbox_y+(bbox_height/2)
        
        central_region = [center_x-(central_region_side/2),
                          center_y-(central_region_side/2),
                          central_region_side,
                          central_region_side]
        
        return np.rint(central_region).astype(int)
    
    def draw_heatmaps(self, image:np.ndarray, heatmap:np.ndarray, alpha:float=0.4)-> np.ndarray:
        """
        Overlays a heatmap on an image using alpha blending.

        This function visualizes model outputs or attention regions by blending a heatmap 
        with the original image. The heatmap is first scaled to 0–255, colorized using the 
        JET colormap for better visual perception, and then blended with the input image 
        using a specified transparency level (`alpha`). The resulting image highlights 
        regions of interest while preserving the original content.

        Args:
            image (np.ndarray): The original input image in BGR format (as loaded by OpenCV).
            heatmap (np.ndarray): A 2D NumPy array representing the heatmap values, 
                expected to be normalized between 0 and 1.
            alpha (float, optional): A blending factor between 0 and 1 that controls the 
                transparency of the heatmap overlay. Default is 0.4.

        Returns:
            np.ndarray: A BGR image of the same shape as `image` with the heatmap blended in.
        """            
        if image is None:
            raise ValueError(f"Could not load the image")

        # Convert heatmap to BGR mask
        heatmap_scaled = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
        
        # Blend image and heatmap
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0.0)
        
        return overlay        


    

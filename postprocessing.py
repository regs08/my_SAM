import cv2
import numpy as np
from segment_anything.utils.amg import remove_small_regions


def extract_segmentation_and_bbox_from_binary_mask(binary_mask, remove_islands=True):
    """
    returns a normilized bbox in yolo format and segmentation [(x,y), (x,y)...]
    """
    if remove_islands:
        binary_mask, _ = remove_small_regions(binary_mask, area_thresh=1000000.0, mode='islands')
        binary_mask = binary_mask.astype(np.uint8)

    h,w = binary_mask.shape

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #normilize the countours
    normed_contours =normalize_segmentation_coords(contours, w,h)
    # Get the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    # Get the new bounding box (x, y, width, height)
    bbox = [int(x) for x in cv2.boundingRect(largest_contour)]
    norm_bbox = normalize_yolo_bbox(bbox, w,h )

    return norm_bbox, normed_contours


def normalize_segmentation_coords(segmentation_coords, image_width, image_height):
    normalized_coords = []
    for coord in segmentation_coords:
        normalized_points = []
        for point in coord:
            x, y = point[0]
            normalized_x = x / image_width
            normalized_y = y / image_height
            normalized_points.append((normalized_x, normalized_y))
        normalized_coords.extend(normalized_points)
    return np.asarray(normalized_coords)


def normalize_yolo_bbox(bbox, image_width, image_height):
    """
    Normalizes YOLO bounding box coordinates.

    Args:
        bbox (tuple): Tuple containing (x, y, width, height) of the bounding box.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        tuple: Tuple containing (x_norm, y_norm, width_norm, height_norm) of the normalized bounding box.
    """
    x, y, width, height = bbox

    # Normalize x, y, width, height by dividing by image width and height
    x_norm = x / float(image_width)
    y_norm = y / float(image_height)
    width_norm = width / float(image_width)
    height_norm = height / float(image_height)

    return x_norm, y_norm, width_norm, height_norm

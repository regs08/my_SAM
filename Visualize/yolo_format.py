import cv2
import numpy as np
import matplotlib.pyplot as plt

from yolo_data.LoadingData.load_utils import get_class_id_bbox_seg_from_yolo


def convert_yolo_to_pascal_voc(yolo_box, image_height, image_width, normilized=True):

  center_x, center_y, width, height = yolo_box
  xmin = int((center_x - width/2) * image_width)
  ymin = int((center_y - height/2) * image_height)
  xmax = int((center_x + width/2) * image_width)
  ymax = int((center_y + height/2) * image_height)
  box = [xmin, ymin, xmax, ymax]
  return box


def plot_image_with_yolo_annotations(image_path, annotation_path):
    """
    Load an image and its annotations (bounding box and segmentation) in YOLO format,
    and plot the image with the annotations overlaid on top.

    Args:
        image_path (str): The path to the input image file.
        annotation_path (str): The path to the annotation file in YOLO format.

    Returns:
        None.
    """
    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    # Create a blank image with the same dimensions as the original image
    masked_image = np.zeros((h, w, 3), dtype=np.uint8)



    # Load the annotations
    with open(annotation_path, 'r') as f:
        for line in f:
            h, w, _ = image.shape
            h = int(h)
            w = int(w)
            annotation = line.strip().split(' ')
            class_label = int(annotation[0])
            bbox_norm = np.array([float(x) for x in annotation[1:5]])
            segmentation_norm = np.array([float(x) for x in annotation[5:]])

            # Convert normalized bbox to pixel coordinates
            x_min, y_min, x_max, y_max = convert_yolo_to_pascal_voc(bbox_norm, image_height=h, image_width=w)

            # Convert normalized segmentation points to pixel coordinates
            segmentation = segmentation_norm.reshape(-1, 2) * np.array([w, h])
            segmentation = segmentation.astype(np.int32)

            # Create a binary mask from the segmentation points
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [segmentation], 255)

            # Set the color and transparency of the mask
            mask_color = (255, 0, 0)
            mask_alpha = 0.5

            # Apply the color and transparency to the mask
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = mask.astype(np.float32) / 255.0
            mask = mask * mask_color
            mask = (mask_alpha * mask).astype(np.uint8)

            # Apply the mask to the blank image
            masked_image = cv2.addWeighted(masked_image, 1, mask, mask_alpha, 0)

            # Draw the bounding box on the image
            cv2.rectangle(masked_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    masked_image = cv2.addWeighted(masked_image, 1, image, mask_alpha, 0)
    # Display the image
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.show()


def plot_image_with_yolo_boxes(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB for Matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    class_ids, bboxes, _ = get_class_id_bbox_seg_from_yolo(label_path)

    # Parse the label data and draw the boxes
    for box in bboxes:
        x_center, y_center, width, height = box
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Show the image with boxes using Matplotlib in Google Colab
    plt.imshow(image)
    plt.axis('off')
    plt.show()


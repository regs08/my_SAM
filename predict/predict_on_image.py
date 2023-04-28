from yolo_data.LoadingData.load_utils import get_yolo_bboxes_from_txt_file, get_annotation_path, glob_image_files
from yolo_data.Conversions.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc

from my_SAM.config import ID_TO_LABEL_MAP
from my_SAM.get_point_labels import get_point_labels

import numpy as np
import cv2
import torch
import os


def apply_sam_to_image_folder_with_boxes(img_dir, ann_dir, predictor, point_labels=False):
    """

    :param img_dir: image folder
    :param ann_dir: ann folder
    :param predictor: instance of SAM predictor
    :return: a list of dictionaries containing the masks, class ids, bboxes, and filename for a given image
        adding in bbox here because we have been getting incomplete predictions to increase training accuracy were gonna
        'give' the bbox away
    """
    image_paths = glob_image_files(img_dir)
    output = []

    for img_path in image_paths:
        data = apply_sam_to_image_with_bboxes(img_path, ann_dir, predictor, point_labels)
        output.append(data)

    return output


def apply_sam_to_image_with_bboxes(img_path, ann_dir, sam_predictor, point_labels=False):
    """
    TODO experiment with the point labels (background, foreground)
    takes in a img path loads it, finds its corresponding annotation, loads in the bbox from the annotation, then uses
    SAM to perform instance segmentation. returns a data package with masks, class ids and filename
    :param img_path: path to load image
    :param ann_dir: folder where anns are stored. note filename w/o ext must==image filename
    :param sam_predictor: predictor instance of SAM
    :param point_labels if we are using the point labels for background, 0, foreground 1
    :return: data package with masks, class ids and filename
    """
    #get ann path
    ann_path = get_annotation_path(img_path, ann_dir)

    #loading in bbox and class_ids - assumed that boxes are in yolo -
    bboxes, class_ids = get_yolo_bboxes_from_txt_file(ann_path)
    print(f'num boxes(instances) in file: {len(bboxes)}')

    #loading in labels
    labels = [ID_TO_LABEL_MAP[id] for id in class_ids]

    #extracting background/foreground info and the class label
    labels, background_foreground = get_point_labels(labels)
    print('unique labels: ', np.unique(np.array(labels)))

    #loading in image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #setting the image to the predictor
    sam_predictor.reset_image()
    sam_predictor.set_image(img)

    #converting image to pascal voc - needed by SAM
    p_v_bboxes = convert_yolo_to_pascal_voc(img, bboxes)

    #transforming boxes
    from_file_bboxes = torch.tensor(p_v_bboxes, device=sam_predictor.device)
    transformed_bboxes_from_file = sam_predictor.transform.apply_boxes_torch(from_file_bboxes, img.shape[:2])

    if point_labels:
        point_labels = background_foreground
    else:
        point_labels=None

    #extracting masks
    masks, _, _ = sam_predictor.predict_torch(
      point_coords=None,
      point_labels=point_labels,
      boxes=transformed_bboxes_from_file,
      multimask_output=False,
    )
    data = {
        'masks': masks,
        'class_ids': class_ids,
        'filename': os.path.basename(img_path),
        'bboxes': bboxes
    }

    return data


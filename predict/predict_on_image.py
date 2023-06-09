from my_SAM.config import ID_TO_LABEL_MAP
from my_SAM.utils.loading_utils import glob_image_files, get_annotation_path, get_class_id_bbox_seg_from_yolo
from my_SAM.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc


import numpy as np
import cv2
import torch
import os


def apply_sam_to_image_folder_with_boxes(img_dir, ann_dir, predictor):
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
        data = apply_sam_to_image_with_bboxes(img_path, ann_dir, predictor)
        output.append(data)

    return output


def apply_sam_to_image_with_bboxes(img_path, ann_dir, sam_predictor, id_to_label_map=ID_TO_LABEL_MAP):
    """
    TODO experiment with the point labels (background, foreground)
    takes in a img path loads it, finds its corresponding annotation, loads in the bbox from the annotation, then uses
    SAM to perform instance segmentation. returns a data package with masks, class ids and filename
    :param img_path: path to load image
    :param ann_dir: folder where anns are stored. note filename w/o ext must==image filename
    :param sam_predictor: predictor instance of SAM
    :return: data package with masks, class ids and filename
    """
    #get ann path
    ann_path = get_annotation_path(img_path, ann_dir)

    #loading in bbox and class_ids - assumed that boxes are in yolo -
    class_ids, bboxes, _ = get_class_id_bbox_seg_from_yolo(ann_path)
    # print(f'class_ids in file: {class_ids}')
    # print(f'num boxes(instances) in file: {len(bboxes)}')

    #loading in labels
    labels = [id_to_label_map[id] for id in class_ids]

    # print('unique labels: ', np.unique(np.array(labels)))

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


    #extracting masks
    masks, _, _ = sam_predictor.predict_torch(
      point_coords=None,
      point_labels=None,
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


from yolo_data.LoadingData.load_utils import get_yolo_bboxes_from_txt_file, get_annotation_path, glob_image_files
from yolo_data.default_param_configs import cat_id_map
from yolo_data.Conversions.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc

import numpy as np
import cv2
import torch


def apply_sam_to_image_folder_with_boxes(img_dir, ann_dir, predictor):
    """

    :param img_dir: image folder
    :param ann_dir: ann folder
    :param predictor: instance of SAM predictor
    :return: a list of masks [m, N,C,H,W]; where m is the mask on the image, N is the number of instances, C (?) (h,w) of image
            a list of class_ids that will have the same len as N in the corresponding mask
    """
    image_paths = glob_image_files(img_dir)
    masks = []
    all_class_ids = []
    for img_path in image_paths:
      mask, class_ids = apply_sam_to_image_with_bboxes(img_path, ann_dir, predictor)
      masks.append(mask)
      all_class_ids.append(class_ids)
    return masks, all_class_ids


def apply_sam_to_image_with_bboxes(img_path, ann_dir, sam_predictor):
    """
    TODO experiment with the point labels (background, foreground)
    takes in a img path loads it, finds its corresponding annotation, loads in the bbox from the annotation, then uses
    SAM to perform instance segmentation. returns a tensor mask in the form of (N, C, H,W)
    :param img_path: path to load image
    :param ann_dir: folder where anns are stored. note filename w/o ext must==image filename
    :param sam_predictor: predictor instance of SAM
    :return: torch tensor of mask
    """
    #get ann path
    ann_path = get_annotation_path(img_path, ann_dir)

    #loading in bbox and class_ids - assumed that boxes are in yolo -
    bboxes, class_ids = get_yolo_bboxes_from_txt_file(ann_path)
    print(f'num boxes(instances) in file: {len(bboxes)}')

    #loading in labels
    labels = [cat_id_map[id] for id in class_ids]
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

    #extracting masks
    masks, _, _ = sam_predictor.predict_torch(
      point_coords=None,
      point_labels=None,
      boxes=transformed_bboxes_from_file,
      multimask_output=False,
    )

    return masks, class_ids


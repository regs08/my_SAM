from my_SAM.ConvertToJson.utils import get_coco_json_format, \
    create_category_annotation, create_image_annotation, create_prediction_submask_anns
from my_SAM.config import LABEL_TO_ID_MAP


def get_img_ann_as_coco_from_sam(pred_output: dict, image_id=0):
    """
    takes in a single prediction and returns the img and seg annotation in coco format
    :param pred_output: dict containing the keys: 'masks', 'bboxes', 'filename', 'class_ids'; gotten from method apply SAM on image with boxes
    :param image_id: id of the image default is 0 for testing
    :param label_id_map:
    :return:
    """

    #create image annotation and segmentaion ann from maskoutput
    mask = pred_output['masks']
    _,_,h,w = mask.shape
    img = create_image_annotation(pred_output['filename'], w, h, image_id=image_id)
    anns = create_prediction_submask_anns(mask, cat_ids=pred_output['class_ids'], image_id=image_id)

    return img, anns


def get_coco_format_from_sam(predictions: list, label_id_map=LABEL_TO_ID_MAP):
    """
    takes in a list of dicts gotten from our predictions, creates the coco format with the label id map
    fills in the image and segmentation info returns the coco file
    :param predictions:
    :param label_id_map:
    :return: dict formatted as coco
    """
    coco_format = get_coco_json_format()
    coco_format['categories'] = create_category_annotation(label_id_map)

    all_img_anns = []
    all_seg_anns = []

    for i, pred in enumerate(predictions):
        #gettting the coco formatted image and segmentation(segmentation, bbox area...) info
        img, anns = get_img_ann_as_coco_from_sam(pred, image_id=i)

        all_img_anns.append(img)
        all_seg_anns.extend(anns)

    coco_format['images'] = all_img_anns
    coco_format['annotations'] = all_seg_anns

    return coco_format
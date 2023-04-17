from my_SAM.ConvertToJson.utils import get_coco_json_format, \
    create_category_annotation, create_image_annotation, create_prediction_submask_anns
from my_SAM.config import ID_TO_LABEL_MAP, LABEL_TO_ID_MAP
import os


def convert_sam_predictions_to_coco(image_path, masks, cat_ids: list, image_id=0, cat_id_map=ID_TO_LABEL_MAP):
    """
    here we take in an image path and the masks generated on it b SAM.
    it is assumed that we have one category.
    """
    coco_format = get_coco_json_format()
    coco_format['categories'] = create_category_annotation(LABEL_TO_ID_MAP)

    _,_,h,w = masks.shape
    filename = os.path.basename(image_path)

    img = create_image_annotation(filename, w,h, image_id=image_id)
    anns = create_prediction_submask_anns(masks, cat_ids=cat_ids, image_id=image_id)

    coco_format['images'] = img
    coco_format['annotations'] = anns
    return coco_format



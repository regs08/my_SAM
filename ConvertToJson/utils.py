from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
from PIL import Image
from my_SAM.config import ID_TO_LABEL_MAP
import numpy as np

"""
Our format for our coco json 
"""


def get_coco_json_format():
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format


def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": 'object',
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list


def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images


def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    if len(segmentation) > 0:
        min_x, min_y, max_x, max_y = polygon.bounds
        width = max_x - min_x
        height = max_y - min_y
        bbox = (min_x, min_y, width, height)
        area = polygon.area

        annotation = {
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": bbox,
            "category_id": category_id,
            "id": annotation_id
        }
    else:
        annotation = {}

    return annotation


"""
compiling sub masks into coco json 
"""


def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        if (poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    return polygons, segmentations


def create_prediction_submask_anns(masks, image_id, cat_ids: list, cat_id_map=ID_TO_LABEL_MAP):
    """
    gonna run with one prediction from one image at a time. we will have to extend the lists of images and annotations
    :param pred: prediction from our NN
    :return: annotations
    """
    sub_mask_anns = []
    N,_,h,w = masks.shape

    for i in range(N):
        #taking out a convert('RGB')
        #torch tensor has a wierd channel in the [N,1, H,W], taking out a channel with 0
        cat_id = cat_ids[i]
        sub_mask = Image.fromarray(masks[i][0].cpu().numpy())
        polygons, segmentations = create_sub_mask_annotation(sub_mask)
        multi_poly = MultiPolygon(polygons)
        sub_mask_anns.append(create_annotation_format(multi_poly,
                                                      segmentations,
                                                      image_id,
                                                      category_id=cat_id_map[cat_id],
                                                      annotation_id=i))
    return sub_mask_anns

"""
loading utils for SAM busniess
"""
import os
import glob

from my_SAM.config import image_exts

"""
loading yolo text files
"""


def get_class_id_bbox_seg_from_yolo(txt_path):
    """
    gets each line as a seperate bbox
    :param txt_file: the text file corresponding to the image
    :return: class_id and bbox or class_id, boox, seg
    """
    lines = read_txt_file(txt_path)
    yolo_bboxes, class_ids, segs = convert_text_lines_to_yolo_format(lines)

    return class_ids, yolo_bboxes, segs


def read_txt_file(txt_path):
    txt_file = open(txt_path, "r")
    lines = txt_file.read().splitlines()
    return lines


def convert_text_lines_to_yolo_format(lines):
    bboxes = []
    class_ns = []
    segs = []
    for idx, line in enumerate(lines):
        value = line.split()
        cls = int(value[0])
        x = float(value[1])
        y = float(value[2])
        w = float(value[3])
        h = float(value[4])
        #if we have segmentation data append it
        if len(line) > 5:
            segs.append([float(i) for i in value[5:]])

        bboxes.append([x,y,w,h])
        class_ns.append(cls)

    return bboxes, class_ns, segs


"""
###
###
"""

def get_annotation_path(image_path, ann_dir):
    """
    Given an image path and an annotation directory, return the annotation file path
    with the same name as the image file but with a .txt extension.

    Args:
        image_path (str): The path to the image file.
        ann_dir (str): The directory where the annotation file should be saved.

    Returns:
        The annotation file path as a string.
    """
    basename = os.path.basename(image_path)
    annotation_name = os.path.splitext(basename)[0] + '.txt'
    annotation_path = os.path.join(ann_dir, annotation_name)
    assert os.path.exists(annotation_path), f'PATH: {annotation_path} \nDOES NOT EXIST'

    return annotation_path


def glob_image_files(image_folder, exts=image_exts):
    image_paths = []
    for ext in exts:
        # Use glob to search for files with the current extension
        files = glob.glob(os.path.join(image_folder,'*' + ext))
        # Extend the matching_files list with the found file paths
        image_paths.extend(files)

    return image_paths
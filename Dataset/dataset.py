import glob
from my_SAM.predict.predict_on_image import apply_sam_to_image_with_bboxes
from my_SAM.postprocessing import extract_segmentation_and_bbox_from_binary_mask
from my_SAM.config import LABEL_TO_ID_MAP, ID_TO_LABEL_MAP
from yolo_data.WritingRenamingFile.writing_to_file_utils import segmentation_to_yolo_line, write_lines_to_file
from my_SAM.ConvertToJson.convert import get_coco_format_from_sam

from matplotlib import pyplot as plt
import re
import numpy as np
import json
import random
import os
import shutil


class MyDataset():
    def __init__(self,
                image_dir,
                image_exts=['.jpg', '.tiff', '.png', '.jpeg', '.JPG']
                ):
        self.image_dir = image_dir
        self.image_exts = image_exts
        self.image_paths = self.get_image_files()

    def get_image_files(self):
        file_list = []
        for ext in self.image_exts:
            glob_path = os.path.join(self.image_dir, '*' + ext)
            file_list.extend(glob.glob(glob_path))
        return file_list


class mySAMOutput(MyDataset):

    def __init__(self, image_dir, sam_predictor, anns=None, point_labels=False, input_format='yolo', id_to_label_map=ID_TO_LABEL_MAP):
        """

        :param image_dir: image folder
        :param sam_predictor:  SAM predictor instance
        :param anns: ann folder or json
        :param point_labels: weather to use point labels in sam predictions
        :param input_data: yolo or coco
        :param args:
        """
        super().__init__(image_dir=image_dir)

        self.sam_predictor = sam_predictor
        self.anns= anns
        self.point_labels = point_labels
        self.input_format = input_format
        self.image_paths = self.get_image_files()
        self.id_to_label_map = id_to_label_map

    def prepare(self):
        """
        prepares the dataset with the given args. predicts on the images given in the image directory
        :return:
        """
        if self.input_format == 'yolo':
            self.mask_output = self.predict_on_images_yolo_bboxes()
        if self.input_format == 'coco':
            self.mask_output = self.predict_on_images_coco_bboxes()
        else:
            print('invalid mask output')

        self.set_binary_masks()

    def set_binary_masks(self):
      for mask_data in self.mask_output:
        b_masks = []
        for i, mask in enumerate(mask_data['masks']):
          binary_mask = mask.cpu().numpy().squeeze().astype(np.uint8)
          b_masks.append(binary_mask)
          mask_data['b_masks'] = b_masks

    """
    Input format 
    """

    def predict_on_images_yolo_bboxes(self):
        output = []
        for img_path in self.image_paths:
            data = apply_sam_to_image_with_bboxes(img_path, self.anns, self.sam_predictor, self.point_labels,
                                                  id_to_label_map=self.id_to_label_map)
            output.append(data)
        return output

    def predict_on_images_coco_bboxes(self):
        pass

    """
    Output format
    """

    def output_to_yolo(self, remove_islands=False, outfolder=None):
        """
        takes our predctions from mask_output and saves them into a normalized yolo format
        :param remove_islands: remove small islands via SAM
        :param outfolder: save folder
        :return: the save folder
        """
        assert outfolder
        for mask_data in self.mask_output:
            lines = []
            for i, mask in enumerate(mask_data['masks']):

                binary_mask = mask.cpu().numpy().squeeze().astype(np.uint8)
                # returns a norm bbox and seg
                _, seg = extract_segmentation_and_bbox_from_binary_mask(binary_mask, remove_islands=remove_islands)
                bbox = mask_data['bboxes'][i]
                # convert our data into a writable line in yolo format append it line by line
                class_id = mask_data['class_ids'][i]
                # class_id = remove_bckgrnd_marker_from_class_id(class_id)

                lines.append(segmentation_to_yolo_line(class_id=class_id, bbox=bbox, segmentation=seg))

            filename = os.path.splitext(os.path.basename(mask_data['filename']))[0] + '.txt'
            filepath = os.path.join(outfolder, filename)
            write_lines_to_file(filepath, lines)

        return outfolder

    def output_to_coco_json(self, save_path):

        for mask_data in self.mask_output:
            # plus one because background is 0 in coco
            # should take in bboxes from mask out put (more accurate)
            # hacky fix had to change label_map to start with 0 because of COCO
            mask_data['class_ids'] = [remove_bckgrnd_marker_from_class_id(id + 1) for id in mask_data['class_ids']]
        coco_format = get_coco_format_from_sam(self.mask_output)

        with open(save_path, "w") as of:
            json.dump(coco_format, of, indent=4)

        return coco_format

    """
    Vis.
    """

    def plot_image_and_yolo_bboxes(self):
        from my_SAM.Visualize.yolo_format import plot_image_with_yolo_boxes
        assert os.path.exists(self.anns)
        image_path, ann_path = get_random_image_ann_path_from_image_paths(self.image_paths, self.anns)
        plot_image_with_yolo_boxes(image_path, ann_path)

    def plot_image_coco_bboxes(self):
        pass

    def plot_pred(self):
        import cv2
        idx = random.choice(range(len(self.mask_output)))
        filename = self.mask_output[idx]['filename']
        masks = self.mask_output[idx]['masks']
        boxes = self.mask_output[idx]['bboxes']
        image_path = os.path.join(self.image_dir, filename)
        img_arr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        for mask in masks:
            print(type(mask))
            print(mask)
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box in boxes:
            show_box(box.cpu().numpy(), plt.gca())
        plt.axis('off')
        plt.show()
    """
    Upload to Roboflow
    """
    # def upload_to_roboflow(self):
    #     """
    #     gets our roboflow params from our py file. and uploads the images and anns. note Ive had less issues with coco
    #     json. for yolo must be normed coords and was getting strange results
    #     :return:
    #     """
    #     from roboflow import Roboflow
    #     #api key
    #     from my_SAM import robo_flow_params
    #     from yolo_data.RoboFlow.uploading import upload_images_with_json
    #     rf = Roboflow(api_key=robo_flow_params.api_key)
    #     upload_project = rf.workspace().project(robo_flow_params.upload_project_name)
    #     assert save json path
    #     upload_images_with_json(upload_data, save_json_path, upload_project)

#for removing number and whitespace in our maskoutput class id
def remove_bckgrnd_marker_from_class_id(class_id, label_to_id_map=LABEL_TO_ID_MAP, id_to_label_map=ID_TO_LABEL_MAP):
    #get label
    label = id_to_label_map[class_id]
    #remove background marker e.g ' 0'
    label = remove_nums_whitespace(label)
    #get correct class_id
    class_id = label_to_id_map[label]

    return class_id


def remove_nums_whitespace(string):
    pattern = '[0-9\s]+'
    return re.sub(pattern, '', string)


def get_random_image_ann_path_from_image_paths(image_paths, ann_folder):
    image_path = random.choice(image_paths)
    ann_path = os.path.join(ann_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    return image_path, ann_path


def copy_images(src_folder, dest_folder):
    # Get a list of all image files in the source folder
    img_extensions = [".png", ".jpg", ".jpeg"]
    img_files = [f for f in os.listdir(src_folder) if f.lower().endswith(tuple(img_extensions))]

    # Copy each image file from the source folder to the destination folder
    for img_file in img_files:
        src_path = os.path.join(src_folder, img_file)
        dest_path = os.path.join(dest_folder, img_file)
        shutil.copy2(src_path, dest_path)

    print(f"{len(img_files)} image files copied from {src_folder} to {dest_folder}.")

#####
# Vis utils
#####

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
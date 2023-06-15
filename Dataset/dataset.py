#TODO look at inheriting from supervision, loading from coco, plotting grid
import glob
from my_SAM.predict.predict_on_image import apply_sam_to_image_with_bboxes
from my_SAM.postprocessing import extract_segmentation_and_bbox_from_binary_mask
from my_SAM.config import LABEL_TO_ID_MAP, ID_TO_LABEL_MAP
from my_SAM.utils.writing import segmentation_to_yolo_line, write_lines_to_file
from my_SAM.utils.loading_utils import get_random_image_ann_path_from_image_paths
from my_SAM.ConvertToJson.convert import get_coco_format_from_sam

from matplotlib import pyplot as plt
import numpy as np
import json
import random
import os
import cv2

random.seed(42)

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


class SAMOutput(MyDataset):

    def __init__(self, image_dir, sam_predictor, anns=None, input_format='yolo', id_to_label_map=ID_TO_LABEL_MAP):
        """

        :param image_dir: image folder
        :param sam_predictor:  SAM predictor instance
        :param anns: ann folder or json
        :param input_data: yolo or coco
        :param args:
        """
        super().__init__(image_dir=image_dir)

        self.sam_predictor = sam_predictor
        self.anns = anns
        self.input_format = input_format
        self.image_paths = self.get_image_files()
        self.id_to_label_map = id_to_label_map
        self.label_to_id_map = {v: k for k, v in self.id_to_label_map.items()}

        self.coco_id_to_label_map = {key + 1: value for key, value in id_to_label_map.items()}

        self.coco_label_to_id_map = {v: k for k, v in self.coco_id_to_label_map.items()}

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
        self.rand_idx = random.choice(range(0, len(self.mask_output)))
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
            data = apply_sam_to_image_with_bboxes(img_path, self.anns, self.sam_predictor,
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
                print(class_id)
                lines.append(segmentation_to_yolo_line(class_id=class_id, bbox=bbox, segmentation=seg))

            filename = os.path.splitext(os.path.basename(mask_data['filename']))[0] + '.txt'
            filepath = os.path.join(outfolder, filename)
            write_lines_to_file(filepath, lines)

        return outfolder

    def output_to_coco_json(self, save_path, convert_maps_ids_to_coco=False):
        """
        if our label maps start with 0, and therefor have class ids that start with
        0 we will need to increment by1
        """
        for mask_data in self.mask_output:
            # plus one because background is 0 in coco
            # should take in bboxes from mask out put (more accurate)
            # hacky fix had to change label_map to start with 0 because of COCO
            mask_data['class_ids'] = [id + 1 for id in mask_data['class_ids']]
        coco_format = get_coco_format_from_sam(self.mask_output, self.coco_label_to_id_map)

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

    def load_img_from_mask_output(self, filename):
        image_path = os.path.join(self.image_dir, filename)
        img_arr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        return image_rgb

    def plot_from_dataset(self, idx=None):
        if not idx:
            idx = self.rand_idx

        filename = self.mask_output[idx]['filename']
        boxes = self.mask_output[idx]['bboxes']

        image_rgb = self.load_img_from_mask_output(filename)

        print(f'Plotting {filename}....')
        for box in boxes:
            x_center, y_center, width, height = box
            x_min = int((x_center - width / 2) * image_rgb.shape[1])
            y_min = int((y_center - height / 2) * image_rgb.shape[0])
            x_max = int((x_center + width / 2) * image_rgb.shape[1])
            y_max = int((y_center + height / 2) * image_rgb.shape[0])
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Show the image with boxes using Matplotlib in Google Colab
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    def plot_pred(self, idx=None):
        """

        :param idx: default is set in prepare.
        :return:
        """
        if not idx:
            idx = self.rand_idx
        filename = self.mask_output[idx]['filename']
        boxes = self.mask_output[idx]['bboxes']
        masks = self.mask_output[idx]['masks']

        image_rgb = self.load_img_from_mask_output(filename)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis('off')
        plt.show()

    """
    Upload to Roboflow
    """

    @staticmethod
    def upload_to_roboflow_with_json(api_key, project_name, img_folder, json_path):
        """
        note had much more accurate images displauyed in RF when json is used
        :param api_key:
        :param project_name:
        :return:
        """
        from yolo_data.RoboFlow.uploading import upload_images_with_json
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        upload_project = rf.workspace().project(project_name)
        upload_images_with_json(img_folder, json_path, upload_project)


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
import os
import cv2
import yaml
import numpy as np

from my_SAM.utils.loading_utils import get_annotation_path, get_class_id_bbox_seg_from_yolo
from my_SAM.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class ImageProcessorBase:
    def __init__(self, image_folder, label_folder, image_out_folder, label_out_folder, yaml_file, batch_size=None):
        self.label_folder = label_folder
        self.image_folder = image_folder
        self.image_out_folder = image_out_folder
        self.label_out_folder = label_out_folder
        self.batch_size = batch_size

        self.image_gen = self.generate_images()
        self.classes = self.get_classes_from_yaml(yaml_file)
        self.label_id_map, self.id_label_map = self.create_label_id_maps(self.classes)
        self.image_file_paths = self.get_image_file_paths()

    def get_image_file_paths(self):
        return [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith(IMAGE_EXTS)]

    def get_annotation_path(self, img_path):
        label_path = get_annotation_path(img_path, self.label_folder)
        assert os.path.exists(label_path), f"{label_path} NOT FOUND"
        return label_path

    @staticmethod
    def get_class_id_bbox_seg_from_yolo(ann_path):
        return get_class_id_bbox_seg_from_yolo(ann_path)

    @staticmethod
    def load_image(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def convert_yolo_to_pascal_voc(img, bboxes):
        return convert_yolo_to_pascal_voc(img, bboxes)

    @staticmethod
    def get_classes_from_yaml(yaml_file):
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)

        classes = yaml_data["names"]

        return classes

    @staticmethod
    def create_label_id_maps(class_names):
        label_to_id_map = {label: idx for idx, label in enumerate(class_names)}
        id_to_label_map = {idx: label for label, idx in label_to_id_map.items()}

        return label_to_id_map, id_to_label_map

    def generate_images(self):
        num_images = len(self.image_file_paths)
        num_batches = int(np.ceil(num_images / self.batch_size))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size
            batch_image_paths = self.image_file_paths[start_idx:end_idx]
            yield batch_image_paths

        # Handle the remaining images in the last batch
        remaining_images = num_images % self.batch_size
        if remaining_images > 0:
            last_batch_image_paths = self.image_file_paths[-remaining_images:]
            yield last_batch_image_paths

    def write_data_yaml_file(self, outdir, dataset_folder=None):
        """
        writes a yaml file with or without the train/val/test paths via dataset_folder arg
        :param DIRS: dict containing our  train, val, test, paths
        :param class_labels: class labels used for training
        :param outdir: save dir for the yaml file
        :return: yaml path
        """
        if dataset_folder:
            train_dir = os.path.join(dataset_folder, 'train')
            val_dir = os.path.join(dataset_folder, 'val')
            test_dir = os.path.join(dataset_folder, 'test')
            yaml_dict = {'train': train_dir,
                         'val': val_dir,
                         'test': test_dir,
                         'nc': len(self.classes),
                         'names': self.classes}
        else:
            yaml_dict = {'nc': len(self.classes),
                         'names': self.classes}

        yaml_path = os.path.join(outdir, 'data.yaml')
        with open(yaml_path, 'w') as file:
            documents = yaml.dump(yaml_dict, file)
        return yaml_path
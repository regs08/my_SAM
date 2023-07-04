import cv2
import numpy as np
import torch
from my_SAM.utils.loading_utils import get_annotation_path, get_class_id_bbox_seg_from_yolo
from my_SAM.convert_yolo_pascal_voc import convert_yolo_to_pascal_voc
import supervision as sv
import os
import yaml

IMAGE_EXTS=(".jpg", ".jpeg", ".png")


class SAMImageProcessor:

    def __init__(self, sam_predictor, label_folder, image_folder, yaml_file,):
        self.sam_predictor = sam_predictor
        self.label_folder = label_folder
        self.image_folder = image_folder

        self.classes = self.get_classes_from_yaml(yaml_file)
        self.label_id_map, self.id_label_map = self.create_label_id_maps(self.classes)
        self.image_file_paths = self.get_image_file_paths()
        self.ds = self.get_sam_pred_dataset()

    def get_sam_pred_dataset(self):
        images = {}
        anns = {}

        for img_path in self.image_file_paths:
            img_name = os.path.basename(img_path)
            detection = self.apply_sam_to_image_with_bboxes(img_path)
            images[img_name] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            anns[img_name] = detection

        return sv.DetectionDataset(images=images,
                                      annotations=anns,
                                      classes=self.classes)

    def apply_sam_to_image_with_bboxes(self, img_path):
        ann_path = self.get_annotation_path(img_path)
        class_ids, bboxes, _ = self.get_class_id_bbox_seg_from_yolo(ann_path)
        img = self.load_image(img_path)

        self.sam_predictor.reset_image()
        self.sam_predictor.set_image(img)

        p_v_bboxes = self.convert_yolo_to_pascal_voc(img, bboxes)
        from_file_bboxes = torch.tensor(p_v_bboxes, device=self.sam_predictor.device)
        transformed_bboxes_from_file = self.sam_predictor.transform.apply_boxes_torch(
            from_file_bboxes, img.shape[:2]
        )

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_bboxes_from_file,
            multimask_output=False,
        )
        masks = np.squeeze(masks.cpu().numpy(), axis=1)

        detection = sv.Detections(
            xyxy=np.asarray(p_v_bboxes),
            mask=masks,
            class_id=np.asarray(class_ids)
        )

        return detection

    def get_image_file_paths(self):
        return [os.path.join(self.image_folder, f)
                        for f in os.listdir(self.image_folder) if f.endswith(IMAGE_EXTS)]

    def get_annotation_path(self, img_path):
        label_path = get_annotation_path(img_path, self.label_folder)
        assert os.path.exists(label_path), f'{label_path} NOT FOUND'
        return label_path

    def annotate_mask(self, img_name, mask_annotator=None, box_annotator=None):
        """

        :param img_name: basename of the iamge file
        :param mask_annotator: annotator helper from supervision. default values are provided
        :param box_annotator: annotator helper from supervision. default values are provided
        :return: np array with our mask/box annotations
        """
        if not mask_annotator:
            mask_annotator = sv.MaskAnnotator()
        if not box_annotator:
            box_annotator = sv.BoxAnnotator(thickness=5, text_scale=1.0, text_thickness=2)

        frame_with_boxes = box_annotator.annotate(
            scene=self.ds.images[img_name].copy(),
            detections=self.ds.annotations[img_name],
            labels=[self.id_label_map[id] for id in self.ds.annotations[img_name].class_id]
        )

        frame = mask_annotator.annotate(
            scene=frame_with_boxes,
            detections=self.ds.annotations[img_name]
        )
        return frame

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
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)

        classes = yaml_data['names']

        return classes

    @staticmethod
    def create_label_id_maps(class_names):
        label_to_id_map = {label: idx for idx, label in enumerate(class_names)}
        id_to_label_map = {idx: label for label, idx in label_to_id_map.items()}

        return label_to_id_map, id_to_label_map


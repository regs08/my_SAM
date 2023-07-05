import os
import cv2
import numpy as np
import torch
import supervision as sv
from my_SAM.SAMImageProcessor.ImageProcessorBase import ImageProcessorBase

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

class SAMImageProcessor(ImageProcessorBase):
    def __init__(
        self,
        sam_predictor,
        image_folder,
        label_folder,
        image_out_folder,
        label_out_folder,
        yaml_file,
        batch_size=None,
        min_image_area=.001,
        approximation_percentage=.5
    ):
        """
        Initialize the SAMImageProcessor.
        The SAMImageProcessor is responsible for applying the SAM algorithm to a batch of images.
        After each batch the a SV Detection dataset is created. this is used to
        save the data to disk or view the data.


        Args:
            sam_predictor: The SAM predictor object.
            image_folder: The path to the folder containing input images.
            label_folder: The path to the folder containing label annotations.
            image_out_folder: The path to the folder to store output images.
            label_out_folder: The path to the folder to store output annotations.
            yaml_file: The path to the YAML file containing class names.
            batch_size: The number of images in each batch. Defaults to None.
            min_image_area: The minimum image area threshold. Defaults to 0.001.
            approximation_percentage: The approximation percentage for output annotations. Defaults to 0.5.
        """
        super().__init__(
            image_folder,
            label_folder,
            image_out_folder,
            label_out_folder,
            yaml_file,
            batch_size=batch_size
        )
        self.sam_predictor = sam_predictor
        self.min_image_area = min_image_area
        self.approximation_percentage = approximation_percentage

    def apply_sam_to_all_batches(self):
        while True:
            try:
                ds = self.apply_sam_to_image_batch()
            except StopIteration:
                break

    def apply_sam_to_image_batch(self):
        """
        Apply the SAM algorithm to a batch of images.

        Returns:
            ds: The DetectionDataset containing the SAM predictions for the batch of images.
        """
        batch = next(self.image_gen)
        print(f"Applying SAM to {len(batch)} images..")
        ds = self.get_sam_pred_dataset(batch)
        ds.as_yolo(
            images_directory_path=self.image_out_folder,
            annotations_directory_path=self.label_out_folder,
            min_image_area_percentage=self.min_image_area,
            approximation_percentage=self.approximation_percentage,
        )
        return ds

    def get_sam_pred_dataset(self, image_paths):
        """
        Generate a DetectionDataset object for a batch of image paths.

        Args:
            image_paths: A list of image paths.

        Returns:
            ds: The DetectionDataset containing images and annotations for the batch.
        """
        images = {}
        anns = {}

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            detection = self.apply_sam_to_image_with_bboxes(img_path)
            images[img_name] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            anns[img_name] = detection

        return sv.DetectionDataset(
            images=images,
            annotations=anns,
            classes=self.classes
        )

    def apply_sam_to_image_with_bboxes(self, img_path):
        """
        Apply the SAM algorithm to an image with bounding boxes.

        Args:
            img_path: The path to the input image.

        Returns:
            detection: The detection results (bounding boxes and masks) obtained from SAM.
        """
        ann_path = self.get_annotation_path(img_path)
        assert ann_path, f"MISSING ANNOTATION PATH {ann_path}"
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

    def annotate_mask(self, ds, img_name, mask_annotator=None, box_annotator=None):
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
            scene=ds.images[img_name].copy(),
            detections=ds.annotations[img_name],
            labels=[self.id_label_map[id] for id in ds.annotations[img_name].class_id]
        )

        frame = mask_annotator.annotate(
            scene=frame_with_boxes,
            detections=ds.annotations[img_name]
        )
        return frame

"""
loads in point labels, background foreground from our class labels. background is given by a space and 0 after the class
name e.g 'grape 0' and foreground space and 1; 'grape 1'

for setting the point labels var in SAM predict

            point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point

"""

from yolo_data.LoadingData.load_utils import get_yolo_bboxes_from_txt_file




def get_point_labels(class_labels):
    strings_without_num = []
    num_list = []
    for s in class_labels:
        if '0' in s:
            strings_without_num.append(s.replace('0', '').strip())
            num_list.append(0)
        elif '1' in s:
            strings_without_num.append(s.replace('1', '').strip())
            num_list.append(1)
        else:
            strings_without_num.append(s)
            num_list.append(1)
    return strings_without_num, num_list


yolo_txt = "/Volumes/ColesSSD/Pictures/Vineyard/Canopy/Pinot-Noir/IMG_0198.txt"
bboxes, class_ids = get_yolo_bboxes_from_txt_file(yolo_txt)
label_to_id_map, id_to_cat_map = read_classes("/Volumes/ColesSSD/Pictures/Vineyard/Canopy/Pinot-Noir/classes.txt")

labels = [id_to_cat_map[id] for id in class_ids]
labels, background_foreground = get_point_labels(labels)

print(labels, background_foreground)
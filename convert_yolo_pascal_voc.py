
def convert_yolo_to_pascal_voc(image, yolo_box, normilized=True):
    h, w, _ = image.shape
    boxes = []
    for box in yolo_box:
        center_x, center_y, width, height = box
        xmin = int((center_x - width/2) * w)
        ymin = int((center_y - height/2) * h)
        xmax = int((center_x + width/2) * w)
        ymax = int((center_y + height/2) * h)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes



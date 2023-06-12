"""
writing utils for SAM business
"""

"""
yolo format business
"""


def segmentation_to_yolo_line(class_id, bbox, segmentation):
  """
  prepares our segmentation, flattens it, and creates our for a binary  mask
  """
  flat_seg = [float(coord) for tup in segmentation for coord in tup]
  lines = []
  lines.append(yolo_format_line(class_id, bbox, flat_seg))
  return lines


def yolo_format_line(class_label, bbox, segmentation):
    # Extract normalized bounding box coordinates in YOLO format
    #assumes the segmentation is a flattend list [x,y,x1,y1...xn,yn]
    x_center, y_center, width, height = bbox

    # Create YOLO-formatted line
    line = f"{class_label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # Add normalized segmentation data
    segmentation_str = " ".join(str(x) for x in segmentation)
    line += " " + segmentation_str

    return line


def write_lines_to_file(filepath, data):
    """
    writes data to filepath
    data is a list of lists. where the inner list elements represent a line for the file
    """
    with open(filepath, 'w') as f:
        for lines in data:
          for line in lines:
            f.write(line + '\n')

"""
writing utils for SAM business
"""
import os
import shutil

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


"""
copying
"""


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
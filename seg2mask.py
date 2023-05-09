import numpy as np
from skimage.draw import polygon2mask


def seg2mask(segmentation, h, w):
    # Scale the coordinates to match the image size
    x = np.array(segmentation[0::2]) * w
    y = np.array(segmentation[1::2]) * h

    # Convert the coordinates to integer arrays
    x = x.round().astype(np.int32)
    y = y.round().astype(np.int32)

    shape = (h, w)
    mask = polygon2mask(shape, np.stack((y, x), axis=1))

    return mask.astype(np.uint8)

# def filter_array(arr, lower_bound, upper_bound):
#     # Create a boolean mask indicating which elements are within the range
#     mask = np.logical_and(arr >= lower_bound, arr <= upper_bound)
#     # Select only the elements within the range
#     filtered_arr = arr[mask]
#     return filtered_arr, mask
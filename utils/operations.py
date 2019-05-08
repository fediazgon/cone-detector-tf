import numpy as np


def extract_crops(img, crop_height, crop_width, step_vertical=None, step_horizontal=None):
    """
    Extracts crops of (crop_height, crop_width) from the given image. Starting
    at (0,0) it begins taking crops horizontally and, every time a crop is taken,
    the 'xmin' start position of the crop is moved according to 'step_horizontal'.
    If some part of the crop to take is out of the bounds of the image, one last
    crop is taken with crop 'xmax' aligned with the right-most ending of the image.
    After taking all the crops in one row, the crop 'ymin' position is moved in the
     same way as before.

    Args:
        img (ndarray): image to crop.
        crop_height (int): height of the crop.
        crop_width (int): width of the crop.
        step_vertical (int): the number of pixels to move vertically before taking
            another crop. It's default value is 'crop_height'.
        step_horizontal (int): the number of pixels to move horizontally before taking
            another crop. It's default value is 'crop_width'.

    Returns:
         sequence of 2D ndarrays: each crop taken.
         sequence of tuples: (ymin, xmin, ymax, xmax) position of each crop in the
             original image.

    """

    img_height, img_width = img.shape[:2]
    crop_height = min(crop_height, img_height)
    crop_width = min(crop_width, img_width)

    # TODO: pre-allocate numpy array
    crops = []
    crops_boxes = []

    if not step_horizontal:
        step_horizontal = crop_width
    if not step_vertical:
        step_vertical = crop_height

    height_offset = 0
    last_row = False
    while not last_row:
        # If we crop 'outside' of the image, change the offset
        # so the crop finishes just at the border if it
        if img_height - height_offset < crop_height:
            height_offset = img_height - crop_height
            last_row = True
        last_column = False
        width_offset = 0
        while not last_column:
            # Same as above
            if img_width - width_offset < crop_width:
                width_offset = img_width - crop_width
                last_column = True
            ymin, ymax = height_offset, height_offset + crop_height
            xmin, xmax = width_offset, width_offset + crop_width
            a_crop = img[ymin:ymax, xmin:xmax]
            crops.append(a_crop)
            crops_boxes.append((ymin, xmin, ymax, xmax))
            width_offset += step_horizontal
        height_offset += step_vertical
    return np.stack(crops, axis=0), crops_boxes


def get_absolute_boxes(box_absolute, boxes_relative):
    """
    Given a bounding box relative to some image, and a sequence of bounding
    boxes relative to the previous one, this methods transform the coordinates
    of each of the last boxes to the same coordinate system of the former.

    For example, if the absolute bounding box is [100, 100, 400, 500] (ymin, xmin,
    ymax, xmax) and the relative one is [10, 10, 20, 30], the coordinates of the
    last one in the coordinate system of the first are [110, 410, 120, 430].

    Args:
        box_absolute (ndarray): absolute bounding box.
        boxes_relative (sequence of ndarray): relative bounding boxes.

    Returns:
        sequence of ndarray: coordinates of each of the relative boxes in the
            coordinate system of the first one.

    """
    absolute_boxes = []
    absolute_ymin, absolute_xmin, _, _ = box_absolute
    for relative_box in boxes_relative:
        absolute_boxes.append(relative_box + [absolute_ymin, absolute_xmin, absolute_ymin, absolute_xmin])
    return absolute_boxes


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlap_thresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

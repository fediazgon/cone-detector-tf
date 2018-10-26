from __future__ import division

import argparse
import logging.config
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils

logging.config.fileConfig('logging.ini')

FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'

SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


def ispath(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('No such file or directory: ' + path)
    else:
        return path


parser = argparse.ArgumentParser()
parser.add_argument('--image',
                    required=True,
                    dest='image_path',
                    type=ispath,
                    help='Path to the image')
parser.add_argument('--output-dir',
                    required=True,
                    dest='output_dir',
                    type=ispath,
                    help='Directory to save the image with the detections', )
parser.add_argument('-c', '--crop-size', dest='crop_size', type=int,
                    help='Size of (square) crops to divide the image '
                         'before the detection')
args = parser.parse_args()

image_path = args.image_path
output_dir = args.output_dir
crop_size = args.crop_size


def main():
    # Read TensorFlow graph
    detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

    # Read video from disk and count frames
    img = cv2.imread(args.image_path)

    with tf.Session(graph=detection_graph) as sess:

        tic = time.time()

        boxes = []

        if crop_size:
            crop_height = crop_width = crop_size
            crop_step_vertical = crop_step_horizontal = crop_size - 20
            crops, crops_coordinates = ops.extract_crops(
                img, crop_height, crop_width,
                crop_step_vertical, crop_step_horizontal)

            detection_dict = tf_utils.run_inference_for_batch(crops, sess)

            for box_absolute, boxes_relative in zip(
                    crops_coordinates, detection_dict['detection_boxes']):
                boxes.extend(ops.get_absolute_boxes(
                    box_absolute,
                    boxes_relative[np.any(boxes_relative, axis=1)]))

            boxes = np.vstack(boxes)
            boxes = ops.non_max_suppression_fast(
                boxes, NON_MAX_SUPPRESSION_THRESHOLD)
        else:
            detection_dict = tf_utils.run_inference_for_batch(
                np.expand_dims(img, axis=0), sess)
            boxes = detection_dict['detection_boxes']
            boxes = boxes[np.any(boxes, axis=2)]

        boxes_scores = detection_dict['detection_scores']
        boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

        for box, score in zip(boxes, boxes_scores):
            if score > SCORE_THRESHOLD:
                ymin, xmin, ymax, xmax = box
                color_detected_rgb = cv_utils.predominant_rgb_color(
                    img, ymin, xmin, ymax, xmax)
                text = '{:.2f}'.format(score)
                cv_utils.add_rectangle_with_text(
                    img, ymin, xmin, ymax, xmax,
                    color_detected_rgb, text)

        toc = time.time()
        processing_time_ms = (toc - tic) * 1000
        logging.debug('Detected {} objects in {:.2f} ms'.format(
            len(boxes), processing_time_ms))

        input_image_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = '{}-detection.jpg'.format(input_image_filename)
        cv2.imwrite(os.path.join(output_dir, output_filename), img)


if __name__ == '__main__':
    main()

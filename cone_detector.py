from __future__ import division

import logging
import logging.config
import time

import cv2
import numpy as np
import tensorflow as tf

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils

logging.config.fileConfig('logging.ini')

VIDEO_PATH = 'testdata/sample_video.mp4'
FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'

OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


def main():
    # Read TensorFlow graph
    detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

    # Read video from disk and count frames
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # CROP_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # CROP_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tf.Session(graph=detection_graph) as sess:

        processed_images = 0
        while cap.isOpened():

            if DETECT_EVERY_N_SECONDS:
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        processed_images * fps * DETECT_EVERY_N_SECONDS)

            ret, frame = cap.read()
            if ret:
                tic = time.time()

                # crops are images as ndarrays of shape
                # (number_crops, CROP_HEIGHT, CROP_WIDTH, 3)
                # crop coordinates are the ymin, xmin, ymax, xmax coordinates in
                #  the original image
                crops, crops_coordinates = ops.extract_crops(
                    frame, CROP_HEIGHT, CROP_WIDTH,
                    CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)

                # Uncomment this if you also uncommented the two lines before
                #  creating the TF session.
                # crops = np.array([crops[0]])
                # crops_coordinates = [crops_coordinates[0]]

                detection_dict = tf_utils.run_inference_for_batch(crops, sess)

                # The detection boxes obtained are relative to each crop. Get
                # boxes relative to the original image
                # IMPORTANT! The boxes coordinates are in the following order:
                # (ymin, xmin, ymax, xmax)
                boxes = []
                for box_absolute, boxes_relative in zip(
                        crops_coordinates, detection_dict['detection_boxes']):
                    boxes.extend(ops.get_absolute_boxes(
                        box_absolute,
                        boxes_relative[np.any(boxes_relative, axis=1)]))
                boxes = np.vstack(boxes)

                # Remove overlapping boxes
                boxes = ops.non_max_suppression_fast(
                    boxes, NON_MAX_SUPPRESSION_THRESHOLD)

                # Get scores to display them on top of each detection
                boxes_scores = detection_dict['detection_scores']
                boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

                for box, score in zip(boxes, boxes_scores):
                    if score > SCORE_THRESHOLD:
                        ymin, xmin, ymax, xmax = box
                        color_detected_rgb = cv_utils.predominant_rgb_color(
                            frame, ymin, xmin, ymax, xmax)
                        text = '{:.2f}'.format(score)
                        cv_utils.add_rectangle_with_text(
                            frame, ymin, xmin, ymax, xmax,
                            color_detected_rgb, text)

                if OUTPUT_WINDOW_WIDTH:
                    frame = cv_utils.resize_width_keeping_aspect_ratio(
                        frame, OUTPUT_WINDOW_WIDTH)

                cv2.imshow('Detection result', frame)
                cv2.waitKey(1)
                processed_images += 1

                toc = time.time()
                processing_time_ms = (toc - tic) * 1000
                logging.debug(
                    'Detected {} objects in {} images in {:.2f} ms'.format(
                        len(boxes), len(crops), processing_time_ms))

            else:
                # No more frames. Break the loop
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#!/usr/bin/env python

"""
Usage: extract_frames --video_path=<video> --output_dir=<dir>
                      --seconds=<seconds> --height=<height>

The script will extract frames from <video> every <seconds> seconds. If the
argument <height> is provided, it will resize the frame without altering the
aspect ratio. The script uses 'millis' to name each frame when is extracted.

"""

import os
import time
from argparse import ArgumentParser, ArgumentTypeError

import cv2
from tqdm import tqdm


def isfile(x):
    if not os.path.isfile(x):
        raise ArgumentTypeError('The file {} does not exist!'.format(x))
    else:
        return x


def isdir(x):
    if not os.path.isdir(x):
        raise ArgumentTypeError('Could not find folder {}'.format(x))
    else:
        return x


def image_resize(image, height, interpolation=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    return cv2.resize(image, dim, interpolation=interpolation)


parser = ArgumentParser()
parser.add_argument('--video_path', required=True, dest='video_path',
                    type=isfile, help='Path to video')
parser.add_argument('--output_dir', required=True, dest='output_dir',
                    type=isdir, help='Dir to output frames')
parser.add_argument('-s', '--seconds', default=1, dest='seconds', type=float,
                    help='Extract a frame every n seconds')
parser.add_argument('-he', '--height', dest='desired_height', type=int,
                    help='Desired height of the output frames')
args = parser.parse_args()

video_path = args.video_path
output_dir = args.output_dir
seconds = args.seconds
desired_height = args.desired_height

cap = cv2.VideoCapture(video_path)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_offset = fps * seconds
video_length = frame_count / fps
total_frames = int(video_length / seconds)


def next_name():
    return str(int(round(time.time() * 1000)))


print('Processing {}'.format(os.path.basename(video_path)))
with tqdm(total=total_frames, unit='frames') as pbar:
    for i in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_offset))
        ret, frame = cap.read()
        if desired_height:
            frame = image_resize(frame, desired_height)
        frame_path = os.path.join(output_dir, next_name() + '.jpg')
        cv2.imwrite(frame_path, frame)
        pbar.update(1)

#!/usr/bin/env python

"""
Usage: extract_frames_cropped.py --video_path=<video> --output_dir=<dir>
                                 --seconds=<seconds> --crop-size=<cs>
                                 --step-v=<sv> --step-h=<sh>

The script will extract frames from <video> every <seconds> seconds. For each
frame, it takes multiple crops of size <cs>. Starting in the top-left corner,
each frame is scanned using a crop window of size <cs>, with a slide of <sh>
pixels in the horizontal direction and <sv> pixels in the vertical direction.

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


parser = ArgumentParser()
parser.add_argument('--video_path', required=True, dest='video_path',
                    type=isfile, help='Path to video')
parser.add_argument('--output_dir', required=True, dest='output_dir',
                    type=isdir, help='Path to output frames')
parser.add_argument('--crop-size', required=True, dest='crop_size', type=int,
                    help='Desired size of the output crop', metavar='crop size')
parser.add_argument('--step-v', dest='crop_step_v', required=True, type=int,
                    help='Vertical step of crop in the original image',
                    metavar='crop step vertical')
parser.add_argument('--step-h', dest='crop_step_h', required=True, type=int,
                    help='Horizontal step of crop in the original image',
                    metavar='crop step horizontal')
parser.add_argument('-s', '--seconds', default=1, dest='seconds', type=float,
                    help='Extract a frame every n seconds', metavar='seconds')
args = parser.parse_args()

video_path = args.video_path
output_dir = args.output_dir
seconds = args.seconds
crop_size = args.crop_size
crop_step_v = args.crop_step_v
crop_step_h = args.crop_step_h

cap = cv2.VideoCapture(video_path)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_offset = fps * seconds
video_length = frame_count / fps
total_frames = int(video_length / seconds)

ret, frame = cap.read()
height, width = frame.shape[:2]
number_steps_v = int((height - crop_size) / crop_step_v) + 1
number_steps_h = int((width - crop_size) / crop_step_h) + 1
total_crops = total_frames * number_steps_v * number_steps_h

print('Extracting {} crops from {} frames'.format(total_crops, total_frames))
print('Continue? [y/n]')
valid_input = False
while not valid_input:
    choice = raw_input().lower()
    if choice in {'yes', 'ye', 'y'}:
        valid_input = True
    elif choice in {'no', 'n'}:
        exit(0)
    else:
        print('Please respond with "y" or "n"')


def next_name():
    return str(int(round(time.time() * 1000)))


print('Processing {}'.format(os.path.basename(video_path)))
with tqdm(total=total_frames, unit='frames') as pbar_frames:
    for i in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_offset))
        start_height = 0
        for r in range(number_steps_v):
            start_width = 0
            for c in range(number_steps_h):
                crop_id = '_r{}c{}'.format(r, c)
                crop_path = os.path.join(
                    output_dir, next_name() + crop_id + '.jpg')
                crop = frame[start_height:(start_height + crop_size),
                             start_width:(start_width + crop_size)]
                cv2.imwrite(crop_path, crop)
                start_width += crop_step_h
            start_height += crop_step_v

        pbar_frames.update(1)

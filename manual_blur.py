# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-08-03 18:42:33
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-08-03 19:24:27

import time
import click
import glob
import cv2
import os

from src.face_detector import FaceDetector
from src import utils

frame = None 
ix,iy = -1,-1
x1, y1, = -1,-1
drawing = False
blurring_done = False

# define mouse callback function to draw circle
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, x1, y1, drawing, frame, blurring_done

    x1 = x
    y1 = y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        blurring_done = True
        if y < iy and x < ix:
            ROI = frame[y:iy, x:ix]
            blur = cv2.GaussianBlur(ROI, (91,91), 0) 
            frame[y:iy, x:ix] = blur
        else:
            ROI = frame[iy:y, ix:x]
            blur = cv2.GaussianBlur(ROI, (91,91), 0) 
            frame[iy:y, ix:x] = blur


cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_rectangle)

in_folder = './output'

@click.command()
@click.option('-v','--video_source', default='/dev/video0')
@click.option('-c','--confidence', type=float, default=0.5)
def main(video_source, confidence):
    global ix, iy, drawing, frame, blurring_done
    detector = FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=confidence,
                            overlap_thr=0.7)
    
    #video = cv2.VideoCapture(video_source)
    img_files=glob.glob(in_folder+"/*.jpg")

    n_frames = 0
    fps_cum = 0.0
    fps_avg = 0.0
    idx = 0
    while True:
        print('idx = ', idx)
        frame = cv2.imread(img_files[idx])
        blurring_done = False

        start_time = time.perf_counter()
        bboxes, scores = detector.inference(frame)
        end_time = time.perf_counter()

        n_frames += 1
        fps = 1.0 / (end_time - start_time)
        fps_cum += fps
        fps_avg = fps_cum / n_frames

        while True:
            if drawing:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, (ix, iy), (x1, y1), (0, 255, 0), 2)
                cv2.imshow('Image', temp_frame)
            else:
                cv2.imshow('Image', frame)

            k = cv2.waitKey(10) & 0xFF
            if k == ord('q') or k == 27:
                exit()
            elif k == 100 or k == 83:
                if blurring_done:
                    cv2.imwrite(img_files[idx], frame)
                    print('***** Saved file ', os.path.basename(img_files[idx]))
                idx += 1
                break
            elif k == 97 or k == 81:
                if blurring_done:
                    cv2.imwrite(img_files[idx], frame)
                    print('***** Saved file ', os.path.basename(img_files[idx]))
                idx -= 1
                if idx < 0:
                    print('This is the first image...')
                    idx = 0
                break

if __name__ == '__main__':
    main()
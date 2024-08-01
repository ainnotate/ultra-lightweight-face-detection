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
drawing = False


# define mouse callback function to draw circle
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            #cv2.imshow('Image', temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        #cv2.rectangle(frame, (ix, iy),(x, y),(0, 255, 255), 2)
        ROI = frame[iy:y, ix:x]
        blur = cv2.GaussianBlur(ROI, (91,91), 0) 
        frame[iy:y, ix:x] = blur


#cv2.namedWindow("Image")
#cv2.setMouseCallback("Image", draw_rectangle)


in_folder = '/home/ainnotate/san/Aya/id_project/Urudu/docs-20240731T135121Z-001/urudu/'
out_folder = 'output/'

@click.command()
@click.option('-v','--video_source', default='/dev/video0')
@click.option('-c','--confidence', type=float, default=0.9)
def main(video_source, confidence):
    global frame
    detector = FaceDetector(model='model/public/ultra-lightweight-face-detection-rfb-320/FP16/ultra-lightweight-face-detection-rfb-320.xml',
                            confidence_thr=confidence,
                            overlap_thr=0.7)
    
    #video = cv2.VideoCapture(video_source)
    img_files=glob.glob(in_folder+'*.jpg')

    n_frames = 0
    fps_cum = 0.0
    fps_avg = 0.0
    idx = 0
    for img in img_files:
        # ret, frame = video.read()
        # if ret == False:
        #     print("End of the file or error to read the next frame.")
        #     break

        frame = cv2.imread(img)

        start_time = time.perf_counter()
        bboxes, scores = detector.inference(frame)
        end_time = time.perf_counter()

        n_frames += 1
        fps = 1.0 / (end_time - start_time)
        fps_cum += fps
        fps_avg = fps_cum / n_frames

        #frame = utils.draw_boxes_with_scores(frame, bboxes, scores)
        #frame = utils.put_text_on_image(frame, text='FPS: {:.2f}'.format( fps_avg ))

        out_file_name = os.path.basename(img)
        if len(bboxes) >= 1:
            print(out_file_name, ' --> Face detected, blurring')
            x = bboxes[0][0] - 10
            y = bboxes[0][1] - 10
            x1 = bboxes[0][2] + 10
            y1 = bboxes[0][3] + 10
            ROI = frame[y:y1, x:x1]
            blur = cv2.GaussianBlur(ROI, (91,91), 0) 
            frame[y:y1, x:x1] = blur
        else:
            print(out_file_name, ' --> Face not detect')

        cv2.imwrite(out_folder+'/'+out_file_name, frame)


        # while True:
        #     cv2.imshow('Image', frame)
        #     k = cv2.waitKey(100) & 0xFF
        #     #print('kkkkkkkkkkkkkkkkk = ', k)
        #     if k == ord('q') or k == 27:
        #         exit()
        #     elif k == 100:
        #         idx += 1
        #         break
        #     elif k == 97:
        #         idx -= 1
        #         if idx < 0:
        #             idx = 0
        #         break

if __name__ == '__main__':
    main()
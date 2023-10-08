#!/usr/bin/env python3
"""
 Copyright (c) 2020-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

#import importlib
#cv2=importlib.import_module("cv2==4.7.0.68")
import cv2
import argparse
import pafy
print(cv2.__version__)
import logging as log
import numpy as np
from time import perf_counter
import sys
from pathlib import Path
from flask import Flask,render_template,Response,request,jsonify,session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import base64

app=Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socket = SocketIO(app, cors_allowed_origins='*')

from openvino.runtime import Core, get_version

from utils.network_wrappers import MaskRCNN, SemanticSegmentation
from utils.misc import MouseClick, check_pressed_keys

sys.path.append(str(Path(__file__).resolve().parents[0] / 'python'))
sys.path.append(str(Path(__file__).resolve().parents[0] / 'python/openvino/model_zoo'))

import monitors
from images_capture import open_images_capture
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

WINNAME = 'Whiteboard_inpainting_demo'

def expand_mask(detection, d):
    for i in range(len(detection[0])):
        detection[0][i][2] = extend_mask(detection[0][i][2], d)


def extend_mask(mask, d=70):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        for c in contour:
            for x, y in c:
                cv2.circle(mask, (x, y), d, (1, 1, 1), -1)
    return mask


def remove_background(img, kernel_size=(7, 7), blur_kernel_size=21, invert_colors=True):
    bgr_planes = cv2.split(img)
    result = []
    kernel = np.ones(kernel_size, np.uint8)
    for plane in bgr_planes:
        dilated_img = cv2.morphologyEx(plane, cv2.MORPH_OPEN, kernel)
        dilated_img = cv2.dilate(dilated_img, kernel)
        bg_img = cv2.medianBlur(dilated_img, blur_kernel_size)
        if invert_colors:
            diff_img = 255 - cv2.absdiff(plane, bg_img)
        else:
            diff_img = cv2.absdiff(plane, bg_img)
        result.append(diff_img)
    return cv2.merge(result)

gb_args={
    "device": "CPU",
    "input": "",
    "m_i": "intel/instance-segmentation-security-1039.xml",
    "m_s": "",
    "threshold": 0.6
    }
print(gb_args)
def read_video():
    print(gb_args)
    url = gb_args['input']
    online=False
    if 'youtube' in url:
        online=True
    if online:
        print('one \n',url)
        video = pafy.new(url)
        print('two \n',video)
        best = video.getbest(preftype="mp4")
        print('three',best)
        gb_args['input']=best.url
        print(best.url)
        cap = open_images_capture(best.url, False)
        if cap.get_type() not in ('VIDEO', 'CAMERA'):
            raise RuntimeError("The input should be a video file or a numeric camera ID")
        print('2-Video Read successfully')
    else :
        cap = open_images_capture(url, False)
        if cap.get_type() not in ('VIDEO', 'CAMERA'):
            raise RuntimeError("The input should be a video file or a numeric camera ID")
        print('2-Video Read successfully')
    return cap

def video_feed(input):
# Cheking the instances in the args
    if bool(gb_args['m_i']) == bool(gb_args['m_s']):
        raise ValueError('Set up exactly one of segmentation models: '
                         '--m_instance_segmentation or --m_semantic_segmentation')
    print('3-instances loaded')
    labels_dir = Path(__file__).resolve().parents[0] / 'data/dataset_classes'
# Printing the openvino runtime infos and loading the model(version,models used ...)
    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()
    model_path = gb_args['m_i'] if gb_args['m_i'] else gb_args['m_s']
    log.info('Reading model {}'.format(model_path))
    if gb_args['m_i']:
        labels_file = str(labels_dir / 'coco_80cl_bkgr.txt')
        segmentation = MaskRCNN(core, gb_args['m_i'], labels_file,
                                gb_args['threshold'], gb_args['device'])
    elif gb_args['m_s']:
        labels_file = str(labels_dir / 'cityscapes_19cl_bkgr.txt')
        segmentation = SemanticSegmentation(core, gb_args['m_s'], labels_file,
                                            gb_args['threshold'], gb_args['device'])
    log.info('The model {} is loaded to {}'.format(model_path, gb_args['device']))
# initializing variables for applying the model on the video    
    metrics = PerformanceMetrics()
    black_board = False
    frame_number = 0
    key = -1
#    print('8-variables initialized')
# Reading the video frame by frame
    start_time=perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")
# getting frame details
    out_frame_size = (frame.shape[1], frame.shape[0] * 2)
    output_frame = np.full((frame.shape[0], frame.shape[1], 3), 255, dtype='uint8')
# Open the video again while releasing the new opened video
#Apply the model detection and effects on the opened video frame by frame
    while frame is not None:
#Extract segmentation features from frame and saving it to mask
        mask = None
        detections = segmentation.get_detections([frame])
        expand_mask(detections, frame.shape[1] // 27)
        if len(detections[0]) > 0:
            mask = detections[0][0][2]
            for i in range(1, len(detections[0])):
                mask = cv2.bitwise_or(mask, detections[0][i][2])
        if mask is not None:
            mask = np.stack([mask, mask, mask], axis=-1)
        else:
            mask = np.zeros(frame.shape, dtype='uint8')
        clear_frame = remove_background(frame, invert_colors=not black_board)
#Apply mask on the frame and saving it to the output
        output_frame = np.where(mask, output_frame, clear_frame)
        merged_frame = np.vstack([frame, output_frame])
        merged_frame = cv2.resize(merged_frame, out_frame_size)
        print(start_time)
        metrics.update(start_time)
        latency,fps=metrics.get_last()
        ret,jpeg = cv2.imencode('.jpg', merged_frame)
        img64 = base64.b64encode(jpeg).decode('utf-8')
        response={
            'latency':latency,
            'fps':fps,
            'frame':img64
        }
        yield(response)
        frame_number += 1
        start_time=perf_counter()
        frame = cap.read()

@socket.on('connect')
def handel_connect():
	emit('connection',' connexion in progress')
	emit('frame','start streaming')
	print('client connected')
@socket.on('read_video')
def handel_Video(data):
    gb_args['input']=data['input']
    global cap
    cap= read_video()
    print(gb_args)
@socket.on('stream')
def handle_message(data):
	print(data)
	emit('frame',next(video_feed(data)))
	print('frame sent')

@socket.on('disconnect')
def handle_diconnect():
	print('disconnected')
if __name__== '__main__':
	socket.run(app,host='0.0.0.0')#,port=5000,allow_unsafe_werkzeug=True)

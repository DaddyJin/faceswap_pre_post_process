import cv2
import numpy as np
import dlib
import facelib

from core import mathlib
from facelib import FaceType, LandmarksProcessor

import os
from tqdm import tqdm
from pathlib import Path
from FrameInfo import FrameInfo

def extract_face_from_frame(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./weights/shape_predictor_68_face_landmarks.dat")

    faces = detector(img_gray)
    # print(len(faces))  # one face
    landmarks_points = []
    for face in faces:
        landmarks = predictor(img_gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
    points = np.array(landmarks_points, np.int32)

    face = faces[0]
    rect = face.left(), face.top(), face.right(), face.bottom()

    return rect, points


def draw_point(points):
    img = cv2.imread('./images/00001.png')
    for point in points:
        point = tuple(point)
        cv2.circle(img, point, 1, color=(0, 255, 0))
    cv2.imshow('face', img)
    cv2.waitKey(0)

def extract_frames_opencv(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count),  desc='extract frames from video'):
        success_grab = cap.grab()
        success_retrieve, img_raw = cap.retrieve()

        cv2.imwrite(os.path.join(output_path, str('%05d.jpg' % i)), img_raw)

def prepprocess_video_frames(video_metadata_dir):
    frames_dir = os.path.join(video_metadata_dir, 'frames')
    faces_dir = os.path.join(video_metadata_dir, 'faces')
    align_info_dir = os.path.join(video_metadata_dir, 'align_info')
    for frame_name in tqdm(os.listdir(frames_dir), desc='preprocess_video_frames'):
        frame_path = os.path.join(frames_dir, frame_name)

        frame_image = cv2.imread(frame_path)
        rect, image_landmarks = extract_face_from_frame(frame_path)
        image_size = 512

        # adapted from deepfacelabl\manuscripts\Extractor.py
        face_type = FaceType.FaceType.WHOLE_FACE
        image_to_face_mat = LandmarksProcessor.get_transform_mat(image_landmarks, image_size, face_type)

        # for local
        face_image = cv2.warpAffine(frame_image, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4)
        face_image_landmarks = LandmarksProcessor.transform_points(image_landmarks, image_to_face_mat)

        # for global
        landmarks_bbox = LandmarksProcessor.transform_points(
            [(0, 0), (0, image_size - 1), (image_size - 1, image_size - 1), (image_size - 1, 0)], image_to_face_mat, True)

        # store face image
        face_path = os.path.join(faces_dir, frame_name)
        cv2.imwrite(face_path, face_image)

        #store align_info
        frame_path = frame_path.replace('\\', '/')
        pkl_path = os.path.join(align_info_dir, os.path.splitext(frame_name)[0] + '.pkl')
        frameinfo = FrameInfo(pkl_path=pkl_path)
        frameinfo.set_frame_path(frame_path)
        frameinfo.set_landmarks(face_image_landmarks.tolist())
        frameinfo.set_source_rect(rect)
        frameinfo.set_source_landmarks(image_landmarks.tolist())
        frameinfo.set_image_to_face_mat(image_to_face_mat)
        frameinfo.save()

def initialize_video_metadata_dir(video_path):
    video_metadata_dir = video_path.replace('.mp4', '')
    frames_dir = os.path.join(video_metadata_dir, 'frames')
    faces_dir = os.path.join(video_metadata_dir, 'faces')
    align_info_dir = os.path.join(video_metadata_dir, 'align_info')
    merged_dir = os.path.join(video_metadata_dir, 'merged')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)
    os.makedirs(align_info_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    return video_metadata_dir



if __name__ == '__main__':
    video_path = './workspace/data_dst.mp4'

    #0. initialize video_metadata_dir
    # print('initialize video_metadata_dir')
    video_metadata_dir = initialize_video_metadata_dir(video_path)


    #1. extract frames from videos
    frames_dir = os.path.join(video_metadata_dir, 'frames')
    extract_frames_opencv(video_path, frames_dir)

    #2. extract faces from frame and store alignment info
    prepprocess_video_frames(video_metadata_dir)










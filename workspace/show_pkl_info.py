import pickle
import json
import cv2
import os


if __name__ == '__main__':
    pkl_path = './data_dst/align_info/00000.pkl'
    with open(pkl_path, 'rb') as f:
        frame_info_dict = pickle.load(f)
    f.close()
    print(frame_info_dict.keys())
    print(frame_info_dict['frame_path'])
    print(frame_info_dict['landmarks'])
    print(frame_info_dict['source_rect'])
    print(frame_info_dict['source_landmarks'])
    print(frame_info_dict['image_to_face_mat'])



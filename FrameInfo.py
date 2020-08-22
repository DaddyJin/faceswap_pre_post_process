import cv2
import numpy as np
import os
import pickle

class FrameInfo():
    def __init__(self, pkl_path):
        super(FrameInfo, self).__init__()
        self.pkl_path = pkl_path
        self.frame_info_dict = {}


    def get_landmarks(self):            return np.array ( self.frame_info_dict['landmarks'] )
    def set_landmarks(self, landmarks): self.frame_info_dict['landmarks'] = landmarks

    def get_frame_path(self):                  return self.frame_info_dict.get ('frame_path', None)
    def set_frame_path(self, frame_path): self.frame_info_dict['frame_path'] = frame_path

    def get_source_rect(self):              return self.frame_info_dict.get('source_rect', None)
    def set_source_rect(self, source_rect): self.frame_info_dict['source_rect'] = source_rect

    def get_source_landmarks(self):                     return np.array(self.frame_info_dict.get('source_landmarks', None))
    def set_source_landmarks(self, source_landmarks):   self.frame_info_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.frame_info_dict.get('image_to_face_mat', None)
        if mat is not None:
            return np.array(mat)
        return None
    def set_image_to_face_mat(self, image_to_face_mat):   self.frame_info_dict['image_to_face_mat'] = image_to_face_mat

    def save(self):
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self.frame_info_dict,f)
        f.close()

    def load(self):
        with open(self.pkl_path, 'rb') as f:
            self.frame_info_dict = pickle.load(f)
        f.close()


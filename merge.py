import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from train import swap_face_simulate_model
import os
from FrameInfo import FrameInfo
from facelib import LandmarksProcessor

# merge swapped face to original frame

def merge(video_metadata_dir, blend_type='traditional'):
    # blend_type 'traditional' for poisson blending using entire face mask
    # blend_type 'dfdc' for poisson blending using border face mask


    IMG_SIZE = 512
    frames_dir = os.path.join(video_metadata_dir, 'frames')
    faces_dir = os.path.join(video_metadata_dir, 'faces')
    align_info_dir = os.path.join(video_metadata_dir, 'align_info')
    for frame_name in os.listdir(frames_dir):
        # ger frame image align info
        align_info_path = os.path.join(align_info_dir, os.path.splitext(frame_name)[0] + '.pkl')
        frame_info = FrameInfo(pkl_path=align_info_path)
        frame_info.load()

        # ger original frame image
        frame_path = os.path.join(frames_dir, frame_name)
        frame_image_cv = cv2.imread(frame_path)

        # get original frame image face mask(face area 255)
        image_hull_mask = LandmarksProcessor.get_image_hull_mask(frame_image_cv.shape, np.array(frame_info.get_source_landmarks()))

        # get original frame image face inv_mask(face area 0)
        image_hull_mask_inv = cv2.bitwise_not(image_hull_mask)

        # get original frame image face mask border(face area 255)
        image_hull_mask_blur = cv2.GaussianBlur(image_hull_mask, (5, 5), 0)






        # ger original frame image without face
        frame_image_no_face_cv = cv2.bitwise_and(frame_image_cv, frame_image_cv, mask=image_hull_mask_inv)


        # get face image and swap face
        ori_face_path = os.path.join(faces_dir, frame_name)
        swapped_face_images_cv = swap_face_simulate_model(cv2.imread(ori_face_path))


        #trasnform swapped face to fit face in the original frame
        image_to_face_mat = frame_info.get_image_to_face_mat()
        face_to_image_mat = cv2.invertAffineTransform(image_to_face_mat)
        swapped_face_on_ori_frame = cv2.warpAffine(swapped_face_images_cv, face_to_image_mat, (frame_image_cv.shape[1], frame_image_cv.shape[0]), cv2.INTER_LANCZOS4)

        #apply mask to the transformed swapped face on the original frame
        swapped_masked_face_on_ori_frame = cv2.bitwise_and(swapped_face_on_ori_frame, swapped_face_on_ori_frame, mask=image_hull_mask)




        #calculate face center point in original frame
        local_face_rect = [(0, 0), (0, IMG_SIZE - 1), (IMG_SIZE - 1, IMG_SIZE - 1), (IMG_SIZE - 1, 0), (IMG_SIZE//2, IMG_SIZE//2)]
        global_face_rect = tuple(LandmarksProcessor.transform_points(local_face_rect, face_to_image_mat))
        global_face_center = tuple(global_face_rect[-1])
        global_image_center = (frame_image_cv.shape[1]//2, frame_image_cv.shape[0]//2)

        # calculate face mask center point in original frame
        (x, y, w, h) = cv2.boundingRect(image_hull_mask)
        global_face_mask_center = (int((x + x + w) / 2), int((y + y + h) / 2))

        # for dfdc blending -----------------------------------------------------------------

        # get original frame image face mask border(face area 255)
        contours, hierarchy = cv2.findContours(image_hull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_hull_mask_border = np.zeros(image_hull_mask.shape, np.uint8)
        cv2.drawContours(image_hull_mask_border, contours, -1, (255), 1)
        image_hull_mask_border_blur = cv2.GaussianBlur(image_hull_mask_border, (5, 5), 0)

        # get mask difference
        face_mask_sub_contour = image_hull_mask - image_hull_mask_border

        # get face difference
        swapped_face_on_ori_frame_sub_contour = cv2.bitwise_and(swapped_face_on_ori_frame, swapped_face_on_ori_frame,
                                                                mask=face_mask_sub_contour)




        result = None
        if blend_type == 'traditional':
            result = cv2.seamlessClone(src=swapped_face_on_ori_frame, dst=frame_image_cv,
                                          mask=image_hull_mask_blur, p=global_face_mask_center, flags=cv2.NORMAL_CLONE)
        elif blend_type == 'dfdc':
            seamlessclone = cv2.seamlessClone(src=swapped_face_on_ori_frame, dst=frame_image_cv,
                                              mask=image_hull_mask_border_blur, p=global_face_mask_center,
                                              flags=cv2.NORMAL_CLONE)
            face_mask_sub_contour_inv = cv2.bitwise_not(face_mask_sub_contour)
            seamlessclone_noface = cv2.bitwise_and(seamlessclone, seamlessclone, mask=face_mask_sub_contour_inv)
            result = seamlessclone_noface + swapped_face_on_ori_frame_sub_contour
        else:
            assert False, 'not implemented this blend type: {}'.format(blend_type)

        show_image = 1

        if show_image:
            cv2.imshow('seamlessclone', result)
            cv2.waitKey()
            cv2.imshow('frame_image_cv', frame_image_cv)
            cv2.waitKey()
            # cv2.imshow('frame_image_no_face_cv', frame_image_no_face_cv)
            # cv2.waitKey()
            cv2.imshow('swapped_face_on_ori_frame', swapped_face_on_ori_frame)
            cv2.waitKey()
            # cv2.imshow('swapped_masked_face_on_ori_frame', swapped_masked_face_on_ori_frame)
            # cv2.waitKey()
            cv2.imshow('image_hull_mask', image_hull_mask)
            cv2.waitKey()
            cv2.imshow('image_hull_mask_border', image_hull_mask_border)
            cv2.waitKey()
            cv2.imshow('image_hull_mask_border_blur', image_hull_mask_border_blur)
            cv2.waitKey()
            # cv2.imshow('image_hull_mask_inv', image_hull_mask_inv)
            # cv2.waitKey()
        # cv2.imshow('swapped_face_images_cv', swapped_face_images_cv)
        # cv2.waitKey()


        break

if __name__ == '__main__':
    video_path = './workspace/data_dst.mp4'
    video_metadata_dir = video_path.replace('.mp4', '')
    merge(video_metadata_dir, blend_type='traditional')
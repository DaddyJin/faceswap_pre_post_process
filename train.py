import cv2
import numpy as np

# train faceswap model
# here we show a demo model for test inference

#wanna change dst face to src face
def swap_face_simulate_model(dst_face):
    points = np.array([[128,128], [200,200], [256,256]])
    for point in points:
        point = tuple(point)
        cv2.circle(dst_face, point, 10, color=(0, 255, 0))
    
    swapped_image_path = 'workspace/swap.png'
    dst_face = cv2.imread(swapped_image_path)

    return dst_face

if __name__ == '__main__':
    example_face_image_path = './workspace/data_dst/faces/00000.jpg'
    swapped = swap_face_simulate_model(cv2.imread(example_face_image_path))

    cv2.imshow('1', swapped)
    cv2.waitKey(0)
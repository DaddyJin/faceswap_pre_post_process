# faceswap_pre_post_process

We aim to accomplish the faceswap procedure, regardless of the model training and model inferring。

## Preprocess
Preprocess includes extract frames from videos, extract face from frame. 
**Landmark** detected is in the global space(the original frame). 
We calculate a **image_to_face matrix**, which align the landmark to an aligned face.
We store the Landmark and the matrix to **pkl file** and the **aligned face** to disk.


## Model
Consider the situation that we want to put faces in src videos to dst videos.
We assume there exists a face swap model, which takes dstination aligned faces as input and outputs source generated face.

## Postprocess
When we get the generated source face(aligned), we transform back to the original frame(use the inverse of the image_to_face matrix).
Then Possion Blending is applied to blend the generated face and the original frame.

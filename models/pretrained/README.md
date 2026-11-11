# Pretrained face models

Used by `scripts/image_pipeline.py` and `scripts/face_model.py`. Both run through OpenCV (`cv2.FaceDetectorYN`, `cv2.FaceRecognizerSF`); nothing else needs installing.

| File | What it does | Source | License |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | Finds faces and 5 landmarks (eyes, nose, mouth corners) | [OpenCV Zoo: YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| `face_recognition_sface_2021dec.onnx` | Turns an aligned face into a 128-number embedding | [OpenCV Zoo: SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache 2.0 |

They are committed so the project runs from a fresh clone without downloads.

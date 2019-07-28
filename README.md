<h1 align="center">
  <div style="margin:10px;">
    <img src="https://github.com/fediazgon/cone-detector-tf/blob/assets/logo.png?raw=true" alt="project-logo" width="200px">
  </div>
  cone-detector-tf
</h1>

<h4 align="center">
Cone detector trained using the <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow Object Detection API</a>
</h4>

<p align="center">
  <a href="#requirements">Requirements</a> •
  <a href="#usage">Usage</a> •
  <a href="#model">Model</a> •
  <a href="#license">License</a>
</p>

![project-demo](https://github.com/fediazgon/cone-detector-tf/blob/assets/demo.gif?raw=true?raw=true)

## Requirements

Install the following packages with pip, preferably inside a virtual environment.

```shell
pip install opencv-python
pip install tensorflow
```

## Usage

Run the program with the following command:

```shell
python cone_detector.py
```

Keep in mind that the app takes square crops for each frame in the video before performing the detection. This is because the model used has been trained using a pre-trained model from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) (_ssd_mobilenet_v1_coco_), whose input volume was a square image; and it performs much better after taking square crops.

However, if the video that you use has a different resolution you might want to change the size of the crops using the variables `CROP_WIDTH`, `CROP_SIZE` and `CROP_STEP_X`. If you don't want to take crops (the detection it is much faster but less accurate), uncomment the following lines:

```python
# CROP_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# CROP_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
...
# crops = np.array([crops[0]])
# crops_coordinates = [crops_coordinates[0]]
```

## Model

The model was trained using arround 200 images (70% train and 30% for validation) extracted from a longer version of the sample video, so it might perform worse on different images. If you want to finetune this model, you can find the `.ckpt` files in the `model` folder.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


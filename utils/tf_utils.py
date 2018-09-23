import numpy as np
import tensorflow as tf


def load_model(path_to_frozen_graph):
    """
    Loads a TensorFlow model from a .pb file containing a frozen graph.

    Args:
        path_to_frozen_graph (str): absolute or relative path to the .pb file.

    Returns:
        tf.Graph: a TensorFlow frozen graph.

    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def run_inference_for_batch(batch, session):
    """
    Forward propagates the batch of images in the given graph.

    Args:
        batch (ndarray): (n_images, img_height, img_width, img_channels).
        graph (tf.Graph): TensorFlow frozen graph.
        session (tf.Session): TensorFlow session.

    Returns:
        a dictionary with the following keys:
        num_detections  --  number of detections for each image.
            An ndarray of shape (n_images).
        detection_boxes --  bounding boxes (ymin, ymax, xmin, xmax) for each image.
            An ndarray of shape (n_images, max_detections, 4).
        detection_scores -- scores for each one of the previous detections.
            An ndarray of shape (n_images, max_detections)

    """
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_scores', 'detection_boxes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    # Run inference
    output_dict = session.run(tensor_dict, feed_dict={image_tensor: batch})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = output_dict['num_detections'].astype(np.int)
    img_height, img_width = batch.shape[1:3]
    # Transform from relative coordinates in the image (e.g., 0.42) to pixel coordinates (e.g., 542)
    output_dict['detection_boxes'] = (output_dict['detection_boxes'] * [img_height, img_width,
                                                                        img_height, img_width]).astype(np.int)
    return output_dict

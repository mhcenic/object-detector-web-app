import unittest
import numpy as np
import tensorflow as tf

from src.keras_yolo import yolo_eval
from src.yolo_utils import (get_classes, get_anchors, get_colors_for_classes)


class TestYOLODetector(unittest.TestCase):

    def test_read_classes(self):
        classes_path = '../object-detector-web-app/model_data/coco_classes.txt'
        num_of_classes = 80
        class_names = get_classes(classes_path)
        self.assertTrue(len(class_names) == num_of_classes)
        self.assertEqual(class_names[0], 'person')

    def test_read_anchors(self):
        anchors_path = '../object-detector-web-app/model_data/yolo_anchors.txt'
        anchors = get_anchors(anchors_path)
        array = np.array(
            [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])
        array1 = np.array(
            [[0.57273, 1.677385], [1.87446, 0.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])
        self.assertTrue((anchors == array).all())
        self.assertFalse((anchors == array1).all())

    def test_generate_colors(self):
        class_names = 'person'
        colors = get_colors_for_classes(class_names)
        colors_list = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
        self.assertEqual(colors, colors_list)

    def test_yolo_eval(self):
        with tf.Session() as test_b:
            yolo_outputs = (tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                            tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                            tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
                            tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))

            boxes, scores, classes = yolo_eval(yolo_outputs, image_shape=(720., 1280.))
            self.assertEqual(str(scores[2].eval()), str(138.79124))
            self.assertEqual(str(boxes[2].eval()), "[1292.3297  -278.52167 3876.9893  -835.56494]")
            self.assertTrue(str(classes[2].eval()), str(54))
            self.assertTrue(str(scores.eval().shape) == str((10,)))
            self.assertTrue(str(boxes.eval().shape) == str((10, 4)))
            self.assertTrue(str(classes.eval().shape) == str((10,)))


if __name__ == '__main__':
    unittest.main()

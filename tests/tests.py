import unittest
import numpy as np

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


if __name__ == '__main__':
    unittest.main()

#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import os

from keras import backend as K
from keras.models import load_model

from src.keras_yolo import (yolo_eval, yolo_head)
from src.yolo_utils import (get_classes, get_anchors, get_colors_for_classes, draw_boxes, get_image)

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    'image_file',
    help='test image name')
parser.add_argument(
    '-m',
    '--model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model, defaults to model_data/yolo.h5',
    default='model_data/yolo.h5')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to model_data/yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to model_data/coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-t',
    '--test_path',
    help='path to directory of test images, defaults to images/',
    default='images')
parser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to out/',
    default='out')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .6',
    default=.6)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .6',
    default=.6)


def _main(args):
    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)
    image_file = os.path.expanduser(args.image_file)

    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    class_names = get_classes(classes_path)

    anchors = get_anchors(anchors_path)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]

    colors = get_colors_for_classes(class_names)

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    image_data, image = get_image(image_file, test_path, model_image_size)

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    draw_boxes(image, out_classes, out_boxes, out_scores, class_names, colors)

    image.save(os.path.join(output_path, image_file), quality=90)
    sess.close()


if __name__ == '__main__':
    _main(parser.parse_args())

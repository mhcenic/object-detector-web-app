import numpy as np
import colorsys
import random
import os

from keras import backend as K
from PIL import ImageFont, ImageDraw, Image


def get_classes(classes_path):
    """Load classes from file.

    :param classes_path: path to classes directory.
    :return: classes.
    """

    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


def get_anchors(anchors_path):
    """Load anchors from file.

    :param anchors_path: path to anchors directory
    :return: anchors.
    """
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def get_colors_for_classes(classes):
    """Generate colors for bounding boxes.

    :param classes: classes names.
    :return: colors.
    """
    hsv_tuples = [(x / len(classes), 1., 1.)
                  for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def draw_boxes(image, out_classes, out_boxes, out_scores, classes, colors):
    """Draw boxes on image.

    :param image: image file
    :param out_classes: detected classes.
    :param out_boxes: boxes positions.
    :param out_scores: confidence scores.
    :param classes: classes names.
    :param colors: colors for boxes.
    """
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = classes[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw


def get_image(image_name, test_path, model_image_size):
    """Preprocess image.

    :param image_name: test image name.
    :param test_path: path to test image directory.
    :param model_image_size: model image size.
    :return:    image_data : preprocessed image.

                image : original image.
    """
    image = Image.open(os.path.join(test_path, image_name))
    is_fixed_size = model_image_size != (None, None)
    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = image.resize(
            tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data, image


def predict(sess, boxes, scores, classes, yolo_model, image_data, input_image_shape, image, image_file, class_names,
            colors, output_path):
    """
    Find objects on image.
    """
    out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes], feed_dict={yolo_model.input: image_data,
                                                                                       input_image_shape: [
                                                                                           image.size[1],
                                                                                           image.size[0]],
                                                                                       K.learning_phase(): 0})

    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    draw_boxes(image, out_classes, out_boxes, out_scores, class_names, colors)
    image.save(os.path.join(output_path, image_file), quality=90)


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        print('Creating output path {}'.format(output_dir))
        os.mkdir(output_dir)

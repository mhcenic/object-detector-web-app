#! /usr/bin/env python
import os
import glob
import flask
import base64

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from keras import backend as K
from keras.models import load_model

from src.keras_yolo import yolo_eval, yolo_head
from src.yolo_utils import get_image, get_classes, get_anchors, get_colors_for_classes, predict, create_output_dir

CLASSES_DIR = 'model_data/coco_classes.txt'
ANCHORS_DIR = 'model_data/yolo_anchors.txt'
MODEL_DIR = 'model_data/yolo.h5'
IMAGES_DIR = 'images/'
OUTPUT_DIR = 'out/'

TEST_IMAGE_LIST = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(IMAGES_DIR))]
static_image_route = '/static/'

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

class_names = get_classes(CLASSES_DIR)
anchors = get_anchors(ANCHORS_DIR)
colors = get_colors_for_classes(class_names)

create_output_dir(OUTPUT_DIR)
sess = K.get_session()
yolo_model = load_model(MODEL_DIR)
model_image_size = yolo_model.layers[0].input_shape[1:3]
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2,))

app.title = 'DetApp'

app.layout = html.Div(
    style={
        'textAlign': 'center',
        'backgroundColor': '#fafbfa',
        'position': 'fixed',
        'width': '100%',
        'height': '100%',
        'top': '0px',
        'left': '0px',
        'overflowY': 'scroll',
    }, children=[
        dbc.Modal(
            [
                dbc.ModalHeader("DetApp"),
                dbc.ModalBody("""Web app for image object detection using YOLO algorithm.  
                Select test image and set up minimum confidence threshold.
                
                
                Author: mhcenic
                License: check in repo
                Repo: github.com/mhcenic/object-detector-web-app
                
                Test photos belong to Allan Zellener, downloaded from the site: github.com/allanzelener/YAD2K
                """
                              ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-button", className="ml-auto")
                ),
            ],
            id="info-modal",
            style={
                'white-space': 'pre-line'
            }
        ),

        html.Div([
            html.Div([
                html.Br(),
                html.Img(id='image0',
                         style={'width': '80%'}
                         ),
                html.Br(),
                html.Div(children=["Minimum Confidence Threshold:"],
                         style={'float': 'left',
                                'padding': '0px 10px 10px 20px',
                                'marginTop': 55
                                }
                         ),
                html.Div([
                    dcc.Slider(id='my-slider',
                               min=0.2,
                               max=0.8,
                               step=None,
                               marks={
                                   0.2: '20%',
                                   0.3: '30%',
                                   0.4: '40%',
                                   0.5: '50%',
                                   0.6: '60%',
                                   0.7: '70%',
                                   0.8: '80%'
                               },
                               value=0.6,
                               )
                ], style={'padding': '0px 10px 10px 20px',
                          'width': '75%',
                          'float': 'right'}),

                html.Div(children=["Selected Image:"],
                         style={'float': 'left',
                                'padding': '0px 10px 10px 20px',
                                'marginTop': 55}),
                html.Div([
                    dcc.Dropdown(id='image0-dropdown',
                                 options=[{'label': i, 'value': i} for i in TEST_IMAGE_LIST],
                                 value=TEST_IMAGE_LIST[0],
                                 )
                ], style={'padding': '0px 10px 10px 20px',
                          'width': '75%',
                          'float': 'right',
                          'marginTop': 35}),
            ], style={'width': '49%',
                      'float': 'left'
                      }
            ),

            html.Div([
                html.Br(),
                html.Img(id='image1',
                         style={
                             'width': '80%'
                         }),
                html.Div([
                    dbc.Button("Learn More", id="info-button", className="mr-1")
                ], style={
                    'textAlign': 'right',
                    'marginTop': 100
                })
            ], style={'width': '49%',
                      'float': 'right'
                      })
        ])
    ])


@app.callback(
    dash.dependencies.Output("info-modal", "is_open"),
    [dash.dependencies.Input("info-button", "n_clicks"), dash.dependencies.Input("close-button", "n_clicks")],
    [dash.dependencies.State("info-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    dash.dependencies.Output('image0', 'src'),
    [dash.dependencies.Input('image0-dropdown', 'value')])
def update_image_src(test_image):
    """
    Shows selected test image
    :param test_image: selected test image name
    :return: path to selected test image
    """
    return static_image_route + test_image


# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.jpg'.format(static_image_route))
def serve_image(image_path):
    """
    Helper function to load test images
    :param image_path: path to test images
    :return: test image file
    """
    image_name = '{}.jpg'.format(image_path)
    if image_name not in TEST_IMAGE_LIST:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(IMAGES_DIR, image_name)


@app.callback(
    dash.dependencies.Output('image1', 'src'),
    [dash.dependencies.Input('image0-dropdown', 'value'),
     dash.dependencies.Input('my-slider', 'value')])
def run_script(test_image, slider):
    """
    Runs YOLO detector with params.
    :param test_image: selected test image name
    :param slider: minimum confidence threshold
    :return: image with found objects
    """
    boxes, scores, classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=slider, iou_threshold=.6)
    image_data, image = get_image(test_image, IMAGES_DIR, model_image_size)
    predict(sess, boxes, scores, classes, yolo_model, image_data, input_image_shape, image, test_image, class_names,
            colors, OUTPUT_DIR)

    encoded_image = base64.b64encode(open('out/' + str(test_image), 'rb').read())
    return 'data:image/jpg;base64,{}'.format(encoded_image.decode())


if __name__ == '__main__':
    app.run_server(debug=False, threaded=False)

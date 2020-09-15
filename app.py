import dash
import dash_html_components as html
import dash_core_components as dcc

import os
import glob
import flask
import base64

image_directory = 'images/'
test_image_list = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory))]
static_image_route = '/static/'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.title = 'DetApp'

app.layout = html.Div(
    style={
        # 'verticalAlign':'middle',
        'textAlign': 'center',
        'backgroundColor': '#fafbfa',
        'position': 'fixed',
        'width': '100%',
        'height': '100%',
        'top': '0px',
        'left': '0px',
        'overflowY': 'scroll',
        'font-family': 'calibri'
    }, children=[
        html.Div([
            html.Br(),
            html.Img(id='image0',
                     style={'height': '85%',
                            'width': '85%'
                            }),
            html.Br(),
            html.Div(children=["Minimum Confidence Threshold:"],
                     style={'float': 'left',
                            'padding': '0px 10px 10px 20px',
                            'marginTop': 55}),
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
                             options=[{'label': i, 'value': i} for i in test_image_list],
                             value=test_image_list[0],
                             )
            ], style={'padding': '0px 10px 10px 20px',
                      'width': '75%',
                      'float': 'right',
                      'marginTop': 35})
        ], style={'width': '49%',
                  'float': 'left'
                  }),

        html.Div([
            html.Br(),
            html.Img(id='image1',
                     style={'height': '85%',
                            'width': '85%'
                            })
        ], style={'width': '49%',
                  'float': 'right'
                  })
    ])


@app.callback(
    dash.dependencies.Output('image0', 'src'),
    [dash.dependencies.Input('image0-dropdown', 'value')])
def update_image_src(test_image):
    """
    Shows selected test image.
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
    Helper function to load test images.
    :param image_path: path to test images
    :return: test image file
    """
    image_name = '{}.jpg'.format(image_path)
    if image_name not in test_image_list:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)


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
    os.system('python yolo_detector.py {} -s {}'.format(test_image, slider))
    encoded_image = base64.b64encode(open('out/' + str(test_image), 'rb').read())
    return 'data:image/jpg;base64,{}'.format(encoded_image.decode())


if __name__ == '__main__':
    app.run_server(debug=True)

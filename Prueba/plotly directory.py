import dash
import dash_html_components as html
import dash_core_components as dcc
from Prueba import capture_image as cp

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(dcc.Input(id='input-box', type='text')),
    html.Button('Submit', id='button'),
    html.Div(id='container-button-basic', children='Enter a value and press submit')
])

image_count = 1
@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    img = cp.CaptureImage(value, image_count)
    img.create_dir()
    a = img.save_img()
    return a
    # return 'The input value was "{}" and the button has been clicked {} times'.format(
    #     value,
    #     n_clicks
    # )


if __name__ == '__main__':
    app.run_server(debug=True)
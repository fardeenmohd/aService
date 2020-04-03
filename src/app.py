import os

from bottle import route, run, static_file, request, response, post
from model import Model


@route('/static/<filename:path>')
def send_static(filename):
    return static_file(filename, root='../static')


@route('/')
def index():
    return "Wait for it Boy"


@post('/predictStockMarket')
def make_a_prediction():
    try:
        data = request.json
        model = Model(configs=data["configs"] if "config" in data.keys() else None,
                      data_configs=data["data_configs"] if "data_configs" in data.keys() else None)

    except ValueError:
        # if bad request data, return 400 Bad Request
        response.status = 400
        return

    response.headers['Content-Type'] = 'application/json'

    response_data = str(model.predict())
    return response_data


if os.environ.get('APP_LOCATION') == 'heroku':
    run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    run(host='localhost', port=8080, debug=True, reloader=True)

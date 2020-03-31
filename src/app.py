import os
import simplejson as json
from bottle import route, run, static_file, request, response, get, post
# from run import main


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
        # main() #to test if my request can trigger the function called main in run.py

    except ValueError:
        # if bad request data, return 400 Bad Request
        response.status = 400
        return

    response.headers['Content-Type'] = 'application/json'

    return json.dumps({'message': "You triggered me " + data["name"]})


if os.environ.get('APP_LOCATION') == 'heroku':
    run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    run(host='localhost', port=8080, debug=True, reloader=True)

import os

from bottle import route, run, request, static_file, view


@route('/static/<filename:path>')
def send_static(filename):
    return static_file(filename, root='../static')


@route('/')
def index():
    return "Hello World"


if os.environ.get('APP_LOCATION') == 'heroku':
    run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    run(host='localhost', port=8080, debug=True, reloader=True)

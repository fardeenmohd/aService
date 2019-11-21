import os

from bottle import route, run, static_file, request
from src.postgres.db import get_users, add_user


@route('/static/<filename:path>')
def send_static(filename):
    return static_file(filename, root='../static')


@route('/')
def index():
    return get_users()


@route('/user/add/<user_name>')
def add_user_from_route():
    name = request.query.get('user_name') or None
    if name is None:
        return "Cannot make user with no name"
    else:
        add_user(name)
        return "User " + name + " added"


if os.environ.get('APP_LOCATION') == 'heroku':
    run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    run(host='localhost', port=8080, debug=True, reloader=True)

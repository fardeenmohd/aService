from os import environ
import re
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base, User


def _create_session():
    db_url = environ.get('DATABASE_URL') or 'postgresql://postgres:3122@localhost:5432/postgres'

    if 'pytest' in sys.argv[0]:
        db_url += '_test'

    if not db_url:
        raise EnvironmentError('Need to set (TEST_DATABASE_URL')

    engine = create_engine(db_url, echo=True)
    Base.metadata.create_all(engine)
    create_session = sessionmaker(bind=engine)
    return create_session()


session = _create_session()


# sql functions

def add_user(name):
    session.add(User(name=name))
    session.commit()


def get_users():
    return session.query(User).order_by(User.name.asc()).all()

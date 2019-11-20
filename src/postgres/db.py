from os import environ
import re
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base


def _create_session():
    db_url = environ.get('DATABASE_URL')

    if 'pytest' in sys.argv[0]:
        db_url += '_test'

    if not db_url:
        raise EnvironmentError('Need to set (TEST_)DATABASE_URL')

    engine = create_engine(db_url, echo=True)
    Base.metadata.create_all(engine)
    create_session = sessionmaker(bind=engine)
    return create_session()


session = _create_session()

# sql functions


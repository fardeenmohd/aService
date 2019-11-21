from sqlalchemy import Column, Sequence, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('id_seq'), primary_key=True)
    name = Column(String(20))

    def __repr__(self):
        return "<User('%d', '%s')>" % (self.id, self.name)

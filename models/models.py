from sqlalchemy import Column, Integer, String, Date, Numeric, ForeignKey
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    userid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    employeeid = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    dob = Column(Date, nullable=False)
    contact_number = Column(String(20), nullable=False)
    email = Column(String(100))

class Expense(Base):
    __tablename__ = "expense"
    expenseid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    userid = Column(Integer, ForeignKey("users.userid"), nullable=False)
    mode_of_payment = Column(String(50), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    credit_or_debit = Column(ENUM('credit', 'debit', name='credit_debit', create_type=False), nullable=False)
    expense_date = Column(Date, nullable=False)
    description = Column(String(255), nullable=False)

class UserAuth(Base):
    __tablename__ = "user_auth"
    authid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    userid = Column(Integer, ForeignKey("users.userid"), nullable=False, unique=True, index=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)   
import os
from dotenv import load_dotenv

load_dotenv()

#database configs
db_name = 'expenses'
username='postgres'
password='admin'
host = 'localhost'

DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql://{username}:{password}@{host}/{db_name}")
# Add other config variables as needed, e.g., SECRET_KEY = os.getenv("SECRET_KEY")
import os
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()
uri = os.getenv("MONGO_URI")
if not uri:
    raise RuntimeError("MONGO_URI not set")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
print("Ping:", client.admin.command("ping"))
print("Databases:", client.list_database_names())

from pymongo import MongoClient
from pymongo.errors import PyMongoError


try:
    mongo_client = MongoClient('mongodb://localhost:27017/',
                     maxPoolSize=50,
                     minPoolSize=1,
                     maxIdleTimeMS=30000,
                     waitQueueTimeoutMS=10000)
    mongo_db = mongo_client["chat"]
    counters_collection = mongo_db["counters"]
    users_collection = mongo_db["user"]
    print("MongoDB connection established")
except Exception as e:
    print("Error connecting to MongoDB:", str(e))


def get_next_sequence_value(sequence_name):
    try:
        sequence_document = counters_collection.find_one_and_update(
            {"_id": sequence_name},
            {"$inc": {"sequence_value": 1}},
            return_document=True,
            upsert=True
        )
        return sequence_document["sequence_value"]
    except PyMongoError as e:
        return 'Error1.1 - ' + str(e)


def get_or_create_user_id(user_name):

    try:
        # Check if user_name exists
        record = users_collection.find_one({"user_name": user_name})
        print(record)
        if record:
            # If user exists, return the user_id
            return record['user_id']
        else:
            # If user does not exist, insert new user and return new user_id
            new_user_id = get_next_sequence_value("user_id")
            users_collection.insert_one({"user_id": new_user_id, "user_name": user_name})
            return new_user_id
    except PyMongoError as e:
        return 'Error1.1 - ' + str(e)




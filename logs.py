from datetime import datetime
from pymongo import MongoClient


class EmptyAIResponse(Exception):
    pass

    #def display(self):
        #print('Empty AI Response')

try:
    mongo_client = MongoClient('mongodb://localhost:27017/',
                     maxPoolSize=50,
                     minPoolSize=1,
                     maxIdleTimeMS=30000,
                     waitQueueTimeoutMS=10000)
    mongo_db = mongo_client["querygpt"]
    query_logs_collection = mongo_db["query_logs"]
    print("MongoDB connection established")
except Exception as e:
    print("Error connecting to MongoDB:", str(e))


def ai_response_json(columns,rows):
    # Get the column names from the cursor description
    ai_response = [dict(zip(columns, row)) for row in rows]
    return ai_response

def ai_response_markdown(_) :
    pass

def seq_generator_mess_id(user):
    pass



def log_exceptions():
    pass


def format_value(value):
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    elif isinstance(value, float):
        return f"{value:.2f}"
    elif isinstance(value, int):
        return f"{value}"
    else:
        return str(clsvalue)


def log_query(user_id,message_id,user_query,ai_response,user_feedback):
    Error_flag = '0'
    if Exceptions == {}:
        Error_flag = 'N'
    else:
        Error_flag = 'Y'

    if user_feedback == None:
        user_feedback = 0

    try:
        query_logs_collection.insert_one({
            "user_id":user_id,
            "message_id":message_id,
            "user_query": user_query,
            "ai_response": ai_response,
            "user_feedback":user_feedback,
            "timestamp": datetime.now()
        })
    except Exception as e:
        print('Error logging query to MongoDB: ', str(e))


def update_user_feedback(message_id, user_feedback):
    try:
            result = query_logs_collection.update_one(
            {"message_id": message_id},
            {"$set": {"user_feedback": user_feedback}}
        )

    except Exception as e:
        print("Error updating user_feedback in MongoDB:", str(e))



#log_query(12,457-777,user_query, ai_generated_sql,columns,row,Exceptions,user_feedback)
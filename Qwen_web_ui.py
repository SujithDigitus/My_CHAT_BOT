# Set to 500 MB
import streamlit as st
from PIL import Image
from docx2pdf import convert
import numpy as np
import pythoncom
from Qwen2_5_rag_chain_version import rag_pipeline_with_prompt,Get_summary
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain.schema import Document
from langchain.vectorstores import Chroma
import shutil
import json
import os
from CLIP_DocumentProcessor import process_file, get_text_from_Pdf
from PPT_Processor import process_file as ppt_process_file
from Video_Processor import process_file as Video_process_file

from logs import log_query,EmptyAIResponse, update_user_feedback
from user_data_pool import get_or_create_user_id
import uuid
from dotenv import load_dotenv
import os
user_name = 'Yoghesh'
user_id=get_or_create_user_id(user_name)
message_id = str(uuid.uuid4())

from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
Chat_Message_History = MongoDBChatMessageHistory(
    session_id="test_session_1",  # Replace with a relevant session ID
    connection_string='mongodb://localhost:27017/',
    database_name="Digi_chat_memory",  # Matches the mongo_db above
    collection_name="User_chat_memory"  # Matches the query_logs_collection above
)

print(
    f"""\n\n
    <Chat_Message_History>\n\n
    {Chat_Message_History}\n\n
    <Chat_Message_History>
    \n\n"""

)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, I am Nora here to help you with questions."),
    ]

if "temp" not in st.session_state:
    st.session_state.temp = []
st.session_state.temp.extend(st.session_state.chat_history)
if len(st.session_state.temp) > 6:
    st.session_state.temp = st.session_state.temp[-6:]

from langchain.prompts import PromptTemplate
def formulate_qery(query,chat_message_history):
    prompt_final = PromptTemplate(
        template="""
                  Given a chat history and the latest user question 
                  which might reference context in the chat history, 
                  formulate a standalone question which can be understood 
                  without the chat history. Do NOT answer the question, 
                  just reformulate it if needed and otherwise return it as is.
                  chat_history: {chat_history}
                  Question: {question}
                  """,
        input_variables=["chat_history", "question",]
    )
    question_chain = prompt_final | ChatOpenAI(model = "gpt-4o-mini", temperature =0) |StrOutputParser()
    print(
        f"""\n\n formulated query {question_chain}
            \n\n"""
    )
    return question_chain.invoke({"question":query,"chat_history":Chat_Message_History})

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

st.set_page_config(page_title="HelpdeskBOT --Otis", page_icon=":books:")
# page_background_img=""""
# <style>
# </style>
# """
# st.markdown(page_background_img,unsafe_allow_html=True)
st.title("DIG-i")


if st.button("Refresh"):
    st.session_state.chat_history = [
        AIMessage(content="Hi, I am Nora here to help you with questions."),
    ]
#THIS WILL USED TO SHOW CASE THE LAST MESSAGES OR COVERSATION :
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("ask your question...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.temp.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI", avatar="🕵️‍♂️"):
        with st.spinner("Wait! Let me think....."):
            print(f'*******************************************************************'
                  f'****************{st.session_state.chat_history}')
            reformulated_query = formulate_qery(user_query, st.session_state.temp)
            #doc_info = get_relevant_doc_summary(reformulated_query)
            response,image_pat = rag_pipeline_with_prompt(user_query, st.session_state.temp)
            if image_pat:
                    if response!="I don't know.":
                            st.markdown(response)
                            Chat_Message_History.add_user_message(user_query)
                            Chat_Message_History.add_ai_message(response)
                            st.session_state.chat_history.append(AIMessage(content=response))
                    count=0
                    for i in image_pat:
                              # Loop through each image path in the list
                            try:
                                if i!='':
                                    image = Image.open(fr'C:\Users\DELL\Desktop\Chatbot\My_chat_bot\{i}')  # Open the image using PIL
                                    if count<2:
                                        st.image(image)
                                count=count+1
                            except Exception as e:
                                st.warning(f"Could not load image {i}: {str(e)}")
            else:
                st.markdown(response)
                Chat_Message_History.add_user_message(user_query)
                Chat_Message_History.add_ai_message(response)
                st.session_state.chat_history.append(AIMessage(content=response))



# Define folder paths
work_folder = r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\work_folder"
archive_folder = r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\archive_folder"

# Ensure folders exist
os.makedirs(work_folder, exist_ok=True)
os.makedirs(archive_folder, exist_ok=True)

# Sidebar setup
with st.sidebar:


    if st.session_state["user_id"] is None:
        user_name = st.text_input("Enter your name")

        if user_name:
            # Save username and create user ID
            st.session_state["user_name"] = user_name
            st.session_state["user_id"] = get_or_create_user_id(user_name)
            st.rerun()  # Force re-run to update UI immediately

    # If user_id is set, show username instead of input box
    else:
        st.success(f"Hello, {st.session_state['user_name']}! ✅") 


    st.subheader("Your Files")
    # File uploader for multiple PDF documents
    pdf_docs = st.file_uploader(
        " 📂 Upload your files here and click on 'Process'",
        accept_multiple_files=True,
        key="pdf_uploader",
        on_change=lambda: st.session_state.pop("pdf_docs", None) 
    )
    agree = st.checkbox("Process the files together")
    # Process button to start processing uploaded files
    if st.button("Process"):
        paths=[]
        path=""
        if pdf_docs:  # Check if any files have been uploaded
            for each_file in pdf_docs:
                file_name = each_file.name
                file_extension = file_name.split('.')[-1].lower()  # Get the file extension (e.g., pdf, ppt, mp4)
                mime_type = each_file.type  # Get MIME type (e.g., 'application/pdf', 'application/vnd.ms-powerpoint', etc.)

                # Check file type based on MIME type or extension
                if 'pdf' in mime_type or file_extension in ['pdf','docx','txt']:
                    if file_extension in ['doc','txt']:    
                        pythoncom.CoInitialize()
                        convert(file_name)
                        work_file_path = os.path.join(work_folder, f"{os.path.splitext(each_file.name)[0]}.pdf")
                        archive_file_path = os.path.join(archive_folder,f"{os.path.splitext(each_file.name)[0]}.pdf")
                        print(f"the work_file_path is {work_file_path}")
                    else:
                        work_file_path = os.path.join(work_folder, each_file.name)
                        archive_file_path = os.path.join(archive_folder, each_file.name)
                    with open(work_file_path, "wb") as f:
                     f.write(each_file.getbuffer())
                    st.write(f"Processing {each_file.name}...")
                    process_file(work_file_path)
                    st.session_state['pdf_upload_status'] = f"Successfully processed and moved {each_file.name}"
                    path=path+fr"{work_file_path}.text"
                    paths.append(path)
#
                    # Move processed PDF to the archive folder
                    shutil.move(work_file_path, archive_file_path)
                    st.success(f"Finished processing {each_file.name}  and moved to archive ✅.")
                    # Process the PDF file here
                elif 'ppt' in mime_type or file_extension in ['ppt', 'pptx']:
                    work_file_path = os.path.join(work_folder, each_file.name)
                    archive_file_path = os.path.join(archive_folder, each_file.name)
                    with open(work_file_path, "wb") as f:
                     f.write(each_file.getbuffer())
                    st.write(f"Processing {each_file.name}...")
                    ppt_process_file(work_file_path)
                    st.session_state['pdf_upload_status'] = f"Successfully processed and moved {each_file.name}"
                    shutil.move(work_file_path, archive_file_path)
                    st.success(f"Finished processing {each_file.name}  and moved to archive ✅.")
                    path=path + fr"{work_file_path}.text"
                    paths.append(path)
# 

                    # Move processed PDF to the archive folder
                    # Process the PPT file here
                elif 'video' in mime_type or file_extension in ['mp4', 'mov', 'avi', 'mkv']:
                    work_file_path = os.path.join(work_folder, each_file.name)
                    archive_file_path = os.path.join(archive_folder, each_file.name)
                    with open(work_file_path, "wb") as f:
                     f.write(each_file.getbuffer())
                    st.write(f"Processing {each_file.name}...\n This may take sometime")
                    Video_process_file(work_file_path)
                    st.session_state['pdf_upload_status'] = f"Successfully processed and moved {each_file.name}"
                    path=path + fr"{work_file_path}.text"
                    paths.append(path)
#
                    # Move processed PDF to the archive folder
                    print(fr"{work_folder}\{os.path.splitext(each_file.name)[0]}.pdf",archive_folder)
                    shutil.move(fr"{work_folder}\{os.path.splitext(each_file.name)[0]}.pdf",archive_folder)
                    st.success(f"Finished processing {each_file.name}  and moved to archive ✅.")
                    st.write(f"Processing Video: {file_name}")
                    # os.remove(fr"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Video_to_audio\{each_file.name}")
                    # Process the video file here
                else:
                    st.write(f"Unsupported file type: {file_name}")
                    # Handle unsupported file types (optional)


if 'pdf_upload_status' in st.session_state:
    text=''
    for path in paths:
        file=os.path.basename(path)
        print(file)
        query= get_text_from_Pdf(fr"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Summary\{file}")
        if agree:
            text+=query
        else:
            summary=Get_summary(query)
            st.markdown(summary)
            Chat_Message_History.add_ai_message(summary)
            st.session_state.chat_history.append(AIMessage(content=summary))
        # Optionally, clear the status after displaying it
            print(st.session_state)
    if agree:
            summary=Get_summary(text)
            st.markdown(summary)
            Chat_Message_History.add_ai_message(summary)
            st.session_state.chat_history.append(AIMessage(content=summary))
        

    del st.session_state['pdf_upload_status']
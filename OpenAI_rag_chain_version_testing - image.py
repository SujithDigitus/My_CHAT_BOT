import torch
import re
import json
from PIL import Image
import numpy as np
import chromadb
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import Ollama
# from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain.chains import LLMChain
from transformers import CLIPModel, CLIPProcessor
from langchain.embeddings.base import Embeddings
# from langchain.retrievers import VectorstoreRetriever
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.retrievers import (ContextualCompressionRetriever, MergerRetriever, )

# DocumentCompressorPipeline

import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env
client = OpenAI()

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import openai
from langchain.retrievers import MultiQueryRetriever

import pprint

# Define the custom prompt template
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # CLIP for both text and image embeddings
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # CLIP processor for images

# Initialize the OpenAI LLM
class CLIPEmbeddings(Embeddings):
    def __init__(self):
        self.model = clip_model
        self.processor = clip_processor

    def embed_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.numpy()

    def embed_documents(self, texts):
        return self.embed_text(texts)

    def embed_query(self, query):
        return self.embed_text([query])[0]

    def embed_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.numpy()

    def embed_images(self, image_paths):
        return [self.embed_image(path) for path in image_paths]
    
# right now not used

# def is_requesting_image(user_query):
#     image_keywords = ["image", "picture", "photo", "visual", "screenshot", "graphic", "diagram", "chart", "illustration", "pic"]
#     action_keywords = ["show me", "display", "draw", "see", "view", "render"]

#     user_query = user_query.lower()

#     # Check if any image-related or action-related keywords are in the query
#     if any(keyword in user_query for keyword in image_keywords) or any(action in user_query for action in action_keywords):
#         return True
#     else:
#         return False

def rag_pipeline_with_prompt(query,chat_history):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Define your prompt template
    template = """
You are a helpful help desk technician named Lilis from Linxens company.
Your task is to answer the user's question based *strictly* and *only* on the provided Context and Images. Ensure every piece of information in the answer is *directly* verifiable within the provided Context or Images. Do not infer, assume, or generate information not explicitly stated.
 
Follow these instructions:
 
1. Read the user's Question carefully to identify the specific information requested.
2. Examine the provided Context and Images thoroughly to locate the relevant information.
3. **For specific step explanations (e.g., "explain the 3rd step", "what is step 6", "explain the third step"):**
    - First, identify which process the user is asking about by checking the most recent relevant process discussed in the chat history.
    - Find ONLY the requested step number from that specific process in the Context.
    - Respond with ONLY that step in the format: "Step X: [exact step text from context]."
    - If there is additional explanation or clarification for this specific step provided immediately after the step in the context, add it as: "This step involves [brief explanation based on context]."
    - If no additional explanation exists in the context for that step, provide only the step text.
    - Do NOT provide the entire process or other steps.
4. **If, for a specific step explanation request, the relevant process cannot be identified from the chat history or current context, or if the requested step number does not exist within any identified process in the Context, apply Rule 7.**
5. **For general process questions (e.g., "How to Approve and Onboard a Vendor?"):**
    - Begin your response with: "To [process name], follow the below steps:"
    - Provide the complete process with all relevant steps in order from the Context.
    - **Crucially, if the process in the Context is presented as a numbered list (e.g., 1., 2., 3.), you MUST replicate that exact numerical list format in your response. Do NOT convert it to bullet points.**
6. **For definition or "what is" questions:**
    - Provide a clear, direct answer based on the information in the Context.
    - If the Context provides additional details or explanation, include them to give a complete understanding.
7. **If the answer cannot be found *****anywhere***** in the provided Context or Images, respond ONLY with: "I do not have information about that in the provided documents."**
    - Do NOT include any emojis, images, or extra formatting.
8. Keep responses clean and text-only - no emojis, special characters, or visual elements.
9. Do not mention that you are using a document or context.
10. Do not include irrelevant information like legal disclaimers, page numbers, etc.
 
Chat History (for contextual understanding only; always prioritize and extract answers *solely* from the provided Context):
{chat_history}
 
Question: {question}
 
Context:
{context}

Images:
{images}
 
Answer:
"""
    prompt_final = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question", "images"]
    )

    # Load ChromaDB client
    persist_directory = r"C:\Users\DELL\Desktop\Chatbot\My_Chat_Bot\VectorDB"
    chroma_client = chromadb.PersistentClient(persist_directory)
    clip_embeddings = CLIPEmbeddings()
    # Retrieve the collection 
    collection = chroma_client.get_collection("multimodel_collection_1",embedding_function=OpenCLIPEmbeddingFunction())
    print(f" \n Query for the collection is {query}\n")
    query_result = collection.query(
    query_texts=query,
    include=["embeddings","distances","documents","metadatas"],
    n_results=5,
    )
    documents = query_result["documents"]
    distances = query_result["distances"]
    image_path_dic=query_result["metadatas"][0]
    print(image_path_dic[0]['image_file'])
    path=image_path_dic[0]['image_file']
    index=image_path_dic[0]['image_ref_num']
    with open(path, 'r') as f:
        image_list = json.load(f)
    image_path=image_list[index]

    # Prepare image information for context
    image_info = ""
    if image_path:
        image_info = "Relevant images are available for this context. The images contain visual information that may be relevant to the query."

    context = "\n".join(documents[0])
    rag_chain = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "images": itemgetter("images")
        }
        | prompt_final  # Pass the custom prompt into the chain
        | llm  # Use the language model for answering
        | StrOutputParser()  # Parse the output
    )
    print(f"context of the query is {context}")
    result = rag_chain.invoke({
        "question": query,
        "context": context,
        "chat_history": chat_history,
        "images": image_info
    })
    return result, image_path

def Get_summary(context):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # # Now, prepare the retriever if necessary (same as before)
    template = """
    Your are providing the summary of the document the user has upload.
    Generate a concise and engaging summary of the provided document as context, focusing on:
        A brief introduction to the subject or product.
        Key topics or features covered in the document with the specifics.
        A closing statement to encourage users to explore or use the content
    Don't include irrelavent information like legal disclaimers, proprietary information, or structural like page number, index etc.  
    If you don't know the answer or it is not present in context provided, just say that you don't know, don't try to 
    make up an answer.
    Do not make assumptions or provide information beyond the given context.
    Context: {context}"""
    prompt_final = PromptTemplate(
        template=template,
        input_variables=["context"]
    )
    rag_chain = (
        {
            "context": itemgetter("context"),
        }
        | prompt_final  # Pass the custom prompt into the chain
        | llm  # Use the language model for answering
        | StrOutputParser()  # Parse the output
    )
    # MergerRetriever can be used if you're combining multiple retrievers
    # # Execute the RAG pipeline with the user query and chat history
    result = rag_chain.invoke({"context":context})

    return result
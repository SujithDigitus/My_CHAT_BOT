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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from transformers import CLIPModel, CLIPProcessor
from langchain.embeddings.base import Embeddings
# from langchain.retrievers import VectorstoreRetriever
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.globals import set_debug
from operator import itemgetter
from langchain.retrievers import (ContextualCompressionRetriever, MergerRetriever, )
from sentence_transformers import CrossEncoder
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# DocumentCompressorPipeline

from langchain.retrievers import MultiQueryRetriever

import pprint

# Define the custom prompt template
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # CLIP for both text and image embeddings
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # CLIP processor for images

# Initialize the Gemini LLM
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
#
#     user_query = user_query.lower()
#
#     # Check if any image-related or action-related keywords are in the query
#     if any(keyword in user_query for keyword in image_keywords) or any(action in user_query for action in action_keywords):
#         return True
#     else:
#         return False


def rag_pipeline_with_prompt(query, chat_history):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # Prompt template
    template = """
    You are a help desk technician called Lilis from Linxens company. You are interacting with a user who is asking you questions about the company's issues. Based on the following user question and context provided, please give detailed answer to the user question.
    don't give any thing apart from answer.
    Don't include irrelavent information like legal disclaimers, proprietary information, or structural like page number, index etc.
    They to be as comprehensive as possible giving well rounded answer
    If you don't know the answer or it is not present in context provided, just say that you don't know, don't try to make up an answer.

    chat_history previous three conversation: {chat_history}

    Question: {question}
    Context: {context}
    
    Answer:"""
    prompt_final = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )

    # Load ChromaDB client
    persist_directory = r"C:\Users\DELL\Desktop\Chatbot\My_Chat_Bot\VectorDB"
    chroma_client = chromadb.PersistentClient(persist_directory)
    # Retrieve the collection 
    collection = chroma_client.get_collection("multimodel_collection_1", embedding_function=OpenCLIPEmbeddingFunction())
    print(f"\nQuery for the collection is {query}\n")

    # Retrieve 30 chunks
    query_result = collection.query(
        query_texts=query,
        include=["embeddings", "distances", "documents", "metadatas"],
        n_results=30,
    )
    documents = query_result["documents"]
    distances = query_result["distances"]
    metadatas = query_result["metadatas"]

    # Rerank with CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    doc_texts = [" ".join(chunk) for chunk in documents]
    pairs = [[query, doc] for doc in doc_texts]
    scores = cross_encoder.predict(pairs)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    documents = [documents[i] for i in ranked_idx]
    distances = [distances[i] for i in ranked_idx]
    metadatas = [metadatas[i] for i in ranked_idx]

    # Take top 5 after reranking
    top_k = 5
    documents = documents[:top_k]
    distances = distances[:top_k]
    metadatas = metadatas[:top_k]

    # Safely extract image path metadata if available
    image_path = None
    if metadatas and isinstance(metadatas[0], list) and metadatas[0] and 'image_file' in metadatas[0][0]:
        image_meta = metadatas[0][0]
        file_list_path = image_meta.get('image_file')
        ref_index = image_meta.get('image_ref_num', 0)
        try:
            with open(file_list_path, 'r') as f:
                image_list = json.load(f)
            image_path = image_list[ref_index]
        except Exception as e:
            print(f"Failed to load image metadata: {e}")
    else:
        print("No valid image metadata found; skipping image retrieval.")

#----------------------------------------Commenting out the image retrieval part-------------------------------#
#    # if is_requesting_image(query)==True:
#    #     query_result_image = collection.query(
#    #     query_embeddings=query_emb,
#    #     include=["embeddings","distances","documents","metadatas"],
#    #     n_results=2,
#    #     )
#    #     metadatas_image = query_result_image["metadatas"]
#    #     ids_images=(query_result_image["ids"])[0]
#    #     pattern = r"(?<=_)(extracted_images\\.*)"
#    #     for i,metadata in enumerate(metadatas_image[0]):
#    # # Check if 'type' key exists and if its value is 'image'
#    #         if "type" in metadata and metadata["type"] == "image":
#    #             match = re.search(pattern, ids_images[i])
#    #             image_path.append(match.group(1))
#    #         else:
#    #             print(None)

    context = "\n".join(documents[0])  # Joining the top documents as the context for the model  # Optionally append image info to the context (or you can process them separately)
    rag_chain = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt_final  # Pass the custom prompt into the chain
        | llm  # Use the language model for answering
        | StrOutputParser()  # Parse the output
    )
    print(f"context of the query is {context}")
    result = rag_chain.invoke({"question": query,"context":context,"chat_history":chat_history})
    return result,image_path # for testing image is not included


def Get_summary(context):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
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
    Context: {context}
    
    Summary:"""
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
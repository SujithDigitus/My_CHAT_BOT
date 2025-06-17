# edited_OpenAI_rag_chain_version_testing_CrossEncoder_Reranking.py
import torch
import re
import json
import base64
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
from openai import OpenAI as OpenAIClient
from langchain.chains import LLMChain
from transformers import CLIPModel, CLIPProcessor
from langchain.embeddings.base import Embeddings
# from langchain.retrievers import VectorstoreRetriever
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.retrievers import (ContextualCompressionRetriever, MergerRetriever, )
from sentence_transformers import CrossEncoder
from langchain_core.output_parsers import StrOutputParser

import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env
client = OpenAIClient()

# DocumentCompressorPipeline

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
#
#     user_query = user_query.lower()
#
#     # Check if any image-related or action-related keywords are in the query
#     if any(keyword in user_query for keyword in image_keywords) or any(action in user_query for action in action_keywords):
#         return True
#     else:
#         return False


def rag_pipeline_with_prompt(query, chat_history, inline_images=None):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt template (slightly adjusted for structured image input)
    template = """
You are a help desk technician called Lilis from Linxens company. You are interacting with a user who is asking you questions about the company's issues. Based on the following user question, context, chat history, and any images provided, please give a detailed answer.
Do not give anything apart from the answer.
Don't include irrelevant information like legal disclaimers, proprietary information, or structural data like page numbers, index etc.
Be as comprehensive as possible giving a well-rounded answer.
If you don't know the answer or it is not present in the context provided, just say that you don't know; don't try to make up an answer.
If any images are included, analyze them thoroughly. Refer to them by filename (e.g., "See error1.png") if relevant to your answer.

Chat history (last turns):
{chat_history}

Question: {question}

Context: {context}
"""
    prompt_final = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )

    # Load ChromaDB client
    persist_directory = r"C:\Users\DELL\Desktop\Chatbot\My_Chat_Bot\VectorDB"
    chroma_client = chromadb.PersistentClient(persist_directory)
    collection = chroma_client.get_collection("multimodel_collection_1", embedding_function=OpenCLIPEmbeddingFunction())
    print(f"\nQuery for the collection is {query}\n")

    # Retrieve text chunks
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
    documents = [documents[i] for i in ranked_idx][:5]
    distances = [distances[i] for i in ranked_idx][:5]
    metadatas = [metadatas[i] for i in ranked_idx][:5]

    # ------------------ Original image-from-metadata code (now commented out) ------------------
    # # Extract image paths from metadata
    # image_files = []
    # if metadatas:
    #     for md_list in metadatas:
    #         for md in md_list:
    #             if md.get("type") == "image" and "image_file" in md:
    #                 image_files.append(md["image_file"])
    # if not image_files:
    #     print("No valid image metadata found; skipping image retrieval.")
    #
    # # Base64-encode each image and build OpenAI-compatible image message parts
    # image_contents = []
    # image_filenames = []
    # for img_path in image_files:
    #     try:
    #         with open(img_path, "rb") as img_f:
    #             b64 = base64.b64encode(img_f.read()).decode("utf-8")
    #         fname = os.path.basename(img_path)
    #         image_contents.append({
    #             "type": "image_url",
    #             "image_url": {"url": f"data:image/{fname.split('.')[-1]};base64,{b64}"}
    #         })
    #         image_filenames.append(fname)
    #     except Exception as e:
    #         print(f"Failed to encode image {img_path}: {e}")

    # ------------------ New inline_images handling ------------------
    image_contents = []
    image_filenames = []
    if inline_images:
        for img in inline_images:
            mime_type = img.get("mime_type")
            b64_data = img.get("base64_data")
            if mime_type and b64_data:
                # Build data URI: "data:<mime_type>;base64,<base64_data>"
                data_uri = f"data:{mime_type};base64,{b64_data}"
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })
                # Use a placeholder filename based on mime type
                extension = mime_type.split("/")[-1]
                image_filenames.append(f"inline_image.{extension}")
            else:
                print("Warning: inline_images entry missing mime_type or base64_data.")
    # If no inline_images provided, image_contents remains empty

    # Build context from top documents
    context = "\n".join([" ".join(doc) for doc in documents])

    # Build final text prompt
    text_prompt = prompt_final.format(
        chat_history="\n".join([str(m) for m in chat_history]),
        context=context,
        question=query
    )

    # Print for debug
    print(f"Context: {context}\nEncoded {len(image_contents)} images.\n")

    # Create multimodal message payload
    message_content = [
        {"type": "text", "text": text_prompt},
        *image_contents  # include inline images if provided
    ]

    # Call the model
    response = llm.invoke([{"role": "user", "content": message_content}])

    # Return the answer plus list of filenames so UI can fetch or display
    return response.content, image_filenames


def Get_summary(context):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    template = """
    You are providing the summary of the document the user has uploaded.
    Generate a concise and engaging summary of the provided document as context, focusing on:
        - A brief introduction to the subject or product.
        - Key topics or features covered in the document with specifics.
        - A closing statement to encourage users to explore or use the content.
    Don't include irrelevant information like legal disclaimers or structural details like page numbers.
    If you don't know the answer or it is not present in the context provided, just say that you don't know; don't try to make up an answer.
    Do not make assumptions or provide information beyond the given context.

    Context: {context}
    """
    prompt_final = PromptTemplate(
        template=template,
        input_variables=["context"]
    )
    rag_chain = (
        {"context": itemgetter("context")}
        | prompt_final
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke({"context": context})
    return result

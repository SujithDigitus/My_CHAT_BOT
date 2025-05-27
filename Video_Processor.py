import os
import math
import chromadb.api
import chromadb
import shutil
import re
import numpy as np
import cv2
import pdfplumber
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
from docx import Document as DocxDocument
from docx.shared import Inches
from transformers import CLIPProcessor, CLIPModel
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torch
from langchain_community.vectorstores.utils import filter_complex_metadata
from docx import Document as DocxDocument
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from Videos_to_pdf import Video_to_pdf
#setting up OpenAI using API
from dotenv import load_dotenv
from langchain.llms import OpenAI
load_dotenv()
client = OpenAI()

class Video_Processor:
    def __init__(self, file_path, true_file_name,persist_directory='persist_chroma_pdf'):
        self.file_path = file_path
        self.true_file_name=true_file_name
        self.persist_directory = persist_directory
        self.embedding = OpenCLIPEmbeddingFunction()

    def process(self,file_name):
        self.subject = file_name
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        output_image_dir = 'extracted_images'
        output_text_file_path=fr"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Summary\{self.true_file_name}.text"
        text_file_for_summary = open(output_text_file_path, 'w',encoding='utf-8')
        
        # persisting and defining the chromadb for storing the embedding through CLIP
        persist_directory= r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\VectorDB"
        Chroma_client = chromadb.PersistentClient(persist_directory)
        collection3=Chroma_client.get_or_create_collection(
        name='multimodel_collection_1',
        embedding_function=self.embedding,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": 100
        })
        metadata=[]
        chunks=[]
        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text=page.extract_text()     
                text_file_for_summary.write(f"Page {page_num + 1}:\n{text}\n\n")
                chunk=self.Text_chunker(text)
                if not os.path.exists(output_image_dir):
                    os.makedirs(output_image_dir)
                images=page.images
                if images:
                    for img_index, img in enumerate(images):
                            image_data = img["stream"].get_data()

                        # Convert image data to a Pillow Image object
                            image = Image.open(BytesIO(image_data))

                        # Generate image filename based on page and image index
                            image_filename = f"{file_name}_page_{page_num + 1}_img_{img_index + 1}.jpg"
                            image_path = os.path.join(output_image_dir, image_filename)

                        # Save the image to the specified directory
                            if image.mode == 'RGBA':
                        # Convert to RGB
                                image = image.convert('RGB')
                            image.save(image_path)
                metadata=metadata+[ {'subject': self.subject,'type':'text','file_type': 'videos','image':image_path} for i in chunk]
                chunks=chunks+chunk
        ids_text=[f"{file_name}_text{id+1}" for id in range(len(chunks))]
        collection3.add(
            documents=chunks,
            ids=ids_text,
            metadatas=metadata
        )



    
    def Text_chunker(self,text):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n"]
        )
        chunks = text_splitter.split_text(text)
        return chunks

#Main function that call PPT_Processor
def process_file(file_path):
    """Process a file (CSV, PDF, etc.) by determining its type and calling the appropriate processor."""
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    file = os.path.splitext(file_name)
    filename = file[0]
    
    if file_extension in ['.mp4', '.mov', '.avi', '.mkv']:
        shutil.move(file_path,r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Video_to_audio")
        file_path=Video_to_pdf(file_name)
        # file_path=r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\work_folder\Inlay_Business.pdf"
        file_name = os.path.basename(file_path)
        processor = Video_Processor(file_path,f"{filename}{file_extension}")
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    processor.process(file_name)
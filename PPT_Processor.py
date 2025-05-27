import os
import math
import chromadb.api
import chromadb
import re
import numpy as np
import cv2
# import pdfplumber
import pythoncom
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
from pptxtopdf import convert
import json
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
from header_Footer_watermark import crop_fixed_header_footer_and_remove_images

#setting up OpenAI using API
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()


class PPT_Processor:
    def __init__(self, file_path,file_extension, persist_directory='persist_chroma_pdf'):
        self.file_path = file_path
        self.file_extension=file_extension
        self.persist_directory = persist_directory
        self.embedding = OpenCLIPEmbeddingFunction()

    def process(self,file_name):
        self.subject = file_name
        doc = fitz.open(self.file_path)
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        output_image_dir = r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\extracted_images"
        output_text_file_path=fr"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Summary\{self.subject}{self.file_extension}.text"
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
        image_match_list=[[None for _ in range(20)] for _ in range(len(doc))]
        image_mapping_file=fr"{output_image_dir}\image_map_{file_name}.json"
        for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text_file_for_summary.write(f"Page {page_num + 1}:\n{text}\n\n")
                chunk=self.Text_chunker(text)
                if not os.path.exists(output_image_dir):
                    os.makedirs(output_image_dir)
                images = page.get_images(full=True)
                if images:
                    for img_index, img in enumerate(images):
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                
                                image = Image.open(BytesIO(image_bytes))
                                if image.mode == "RGBA":
                                    image = image.convert("RGB")
                                
                                image_filename = fr"{file_name}_page_{page_num + 1}_img_{img_index + 1}.jpg"
                                image_path = os.path.join(output_image_dir, image_filename)
                                image.save(image_path)
                                print(image_path)
                                image_match_list[page_num][img_index]=image_path
                metadata=metadata+[ {'subject': self.subject,'type':'text','file_type': 'PPT','image_ref_num':page_num,"image_file":image_mapping_file} for i in chunk]
                chunks=chunks+chunk
        with open(image_mapping_file, 'w') as f:
            json.dump(image_match_list, f)
        ids_text=[f"{file_name}_text{id+1}" for id in range(len(chunks))]
        collection3.add(
            documents=chunks,
            ids=ids_text,
            metadatas=metadata
        )
        print(output_text_file_path)    
        return output_text_file_path

    
    def Text_chunker(self,text):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n"]
        )
        chunks = text_splitter.split_text(text)
        return chunks

def initialize_com():
    # Initialize COM in the current thread
    pythoncom.CoInitialize()

def cleanup_com():
    # Uninitialize COM in the current thread when done
    pythoncom.CoUninitialize()

#Main function that call PPT_Processor
def process_file(file_path):
    """Process a file (CSV, PDF, etc.) by determining its type and calling the appropriate processor."""
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    file = os.path.splitext(file_name)
    filename = file[0]
    path_without_ext = os.path.splitext(file_path)[0]
    
    if file_extension in ['.pptx',".ppt"]:
        initialize_com()
        convert(file_path,file_path.replace(file_name,""))
        cleanup_com()
        file_path=file_path.replace(file_extension,".pdf")
        processor = PPT_Processor(file_path,file_extension)
    elif file_extension in [".pdf"]:
        output_file=f"{path_without_ext}_updated.pdf"
        crop_fixed_header_footer_and_remove_images(file_path,output_file)
        processor = PPT_Processor(file_path,file_extension)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    processor.process(filename)

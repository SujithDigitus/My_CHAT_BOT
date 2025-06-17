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

def get_potential_images_from_metadata(metadatas):
    """Extract all potential image paths from metadata"""
    potential_images = []
    
    for metadata_list in metadatas:
        if isinstance(metadata_list, list):
            for metadata in metadata_list:
                if isinstance(metadata, dict) and 'image_file' in metadata:
                    file_list_path = metadata.get('image_file')
                    ref_index = metadata.get('image_ref_num', 0)
                    
                    try:
                        with open(file_list_path, 'r') as f:
                            image_list = json.load(f)
                        
                        # Get all images for this page/reference
                        if ref_index < len(image_list):
                            page_images = image_list[ref_index]
                            if isinstance(page_images, list):
                                for img_path in page_images:
                                    if img_path and img_path not in potential_images:
                                        potential_images.append(img_path)
                            elif page_images and page_images not in potential_images:
                                potential_images.append(page_images)
                    except Exception as e:
                        print(f"Failed to load image metadata: {e}")
    
    # Filter out None values and limit to reasonable number for processing
    potential_images = [img for img in potential_images if img is not None][:15]  # Increased limit for initial screening
    return potential_images

def check_image_relevance_with_gemini_enhanced(query, image_paths, context=""):
    """Enhanced function using Gemini 2.5 Flash to check which images are relevant and get detailed analysis"""
    if not image_paths:
        return [], {}
    
    try:
        # Initialize Gemini model for vision tasks
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Prepare images for the model in batches to avoid overload
        batch_size = 8  # Process in batches of 8 images
        all_relevant_images = []
        all_image_descriptions = {}
        
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start:batch_start + batch_size]
            images_for_model = []
            valid_image_paths = []
            
            for img_path in batch_paths:
                try:
                    if os.path.exists(img_path):
                        # Load and prepare image
                        pil_image = Image.open(img_path)
                        if pil_image.mode == "RGBA":
                            pil_image = pil_image.convert("RGB")
                        
                        images_for_model.append(pil_image)
                        valid_image_paths.append(img_path)
                    else:
                        print(f"Image file not found: {img_path}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            
            if not images_for_model:
                continue
            
            # Enhanced prompt for comprehensive image analysis
            analysis_prompt = f"""
            You are an expert image analyst helping to determine image relevance for a question-answering system.
            
            USER QUERY: "{query}"
            
            CONTEXT FROM RETRIEVED DOCUMENTS: "{context[:500] if context else 'No additional context provided'}"
            
            I will show you {len(images_for_model)} images. For each image, please:
            
            1. Analyze what the image contains (describe key elements, text, diagrams, charts, etc.)
            2. Determine if it's RELEVANT or NOT_RELEVANT to answering the user's query
            3. If relevant, explain HOW it helps answer the query
            4. Provide the image filename/path if relevant
            
            For each image, respond in this EXACT format:
            IMAGE_X_ANALYSIS:
            - Description: [Brief description of what's in the image]
            - Relevance: [RELEVANT or NOT_RELEVANT]
            - Reasoning: [Why it is or isn't relevant to the query]
            - Filename: [If relevant, provide the filename from the path]
            
            Only mark an image as RELEVANT if it directly helps answer the user's question or provides crucial visual information that supports a comprehensive response.
            
            Analyze the images now:
            """
            
            # Create content list with prompt and images
            content = [analysis_prompt]
            content.extend(images_for_model)
            
            try:
                # Generate response with enhanced error handling
                response = model.generate_content(content)
                response_text = response.text
                
                # Parse the enhanced response
                batch_relevant_images, batch_descriptions = parse_enhanced_gemini_response(
                    response_text, valid_image_paths
                )
                
                all_relevant_images.extend(batch_relevant_images)
                all_image_descriptions.update(batch_descriptions)
                
            except Exception as e:
                print(f"Error in Gemini batch analysis: {e}")
                # Fallback: use simple heuristics for this batch
                all_relevant_images.extend(valid_image_paths[:2])  # Take first 2 as fallback
        
        print(f"Enhanced Gemini analysis result: {len(all_relevant_images)} relevant images out of {len(image_paths)}")
        print(f"Relevant images: {all_relevant_images}")
        
        return all_relevant_images, all_image_descriptions
        
    except Exception as e:
        print(f"Error in enhanced Gemini image relevance check: {e}")
        # Fallback: return first few images if error occurs
        return image_paths[:3] if image_paths else [], {}

def parse_enhanced_gemini_response(response_text, image_paths):
    """Parse the enhanced Gemini response to extract relevant images and descriptions"""
    relevant_images = []
    image_descriptions = {}
    
    try:
        # Split by IMAGE_X_ANALYSIS sections
        sections = re.split(r'IMAGE_\d+_ANALYSIS:', response_text)
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty split
            lines = section.strip().split('\n')
            description = ""
            relevance = ""
            reasoning = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('- Description:'):
                    description = line.replace('- Description:', '').strip()
                elif line.startswith('- Relevance:'):
                    relevance = line.replace('- Relevance:', '').strip().upper()
                elif line.startswith('- Reasoning:'):
                    reasoning = line.replace('- Reasoning:', '').strip()
            
            # Check if this image is relevant
            if 'RELEVANT' in relevance and 'NOT_RELEVANT' not in relevance:
                if i <= len(image_paths):
                    img_path = image_paths[i-1]
                    relevant_images.append(img_path)
                    image_descriptions[img_path] = {
                        'description': description,
                        'reasoning': reasoning
                    }
        
        # Fallback parsing if structured parsing fails
        if not relevant_images:
            relevant_count = response_text.upper().count('RELEVANT') - response_text.upper().count('NOT_RELEVANT')
            if relevant_count > 0:
                # Use regex to find relevant mentions
                relevant_mentions = re.findall(r'IMAGE[_\s]*(\d+)[^:]*:.*?RELEVANT[^N]', response_text, re.IGNORECASE | re.DOTALL)
                for mention in relevant_mentions:
                    idx = int(mention) - 1
                    if 0 <= idx < len(image_paths):
                        relevant_images.append(image_paths[idx])
                
                # If still no specific matches, take first few based on count
                if not relevant_images and relevant_count > 0:
                    relevant_images = image_paths[:min(relevant_count, len(image_paths))]
        
    except Exception as e:
        print(f"Error parsing enhanced Gemini response: {e}")
        # Simple fallback
        if 'RELEVANT' in response_text.upper():
            relevant_images = image_paths[:2]
    
    return relevant_images, image_descriptions

def generate_text_response_with_image_references(query, context, relevant_images, image_descriptions, chat_history):
    """Generate text response that naturally references relevant images by filename"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Prepare the content with text and relevant images
        content_parts = []
        
        # Create a mapping of image filenames for easier reference
        image_filename_map = {}
        for img_path in relevant_images:
            filename = os.path.basename(img_path)
            image_filename_map[filename] = img_path
        
        # Enhanced prompt that encourages natural image references
        enhanced_prompt = f"""
        You are a help desk technician called Lilis from Linxens company. You are interacting with a user who is asking you questions about the company's issues. 
        
        Based on the following user question, context provided, and relevant images (if any), please give a detailed answer to the user question.
        
        IMPORTANT: When relevant images support your answer, naturally mention them in your response by referring to their filenames. For example: "As shown in figure_1.png" or "The diagram in chart_overview.jpg illustrates" or "Refer to image_name.png for details".
        
        Don't give anything apart from the answer.
        Don't include irrelevant information like legal disclaimers, proprietary information, or structural details like page numbers, index etc.
        Try to be as comprehensive as possible giving well-rounded answers.
        If you don't know the answer or it is not present in context provided, just say that you don't know, don't try to make up an answer.
        
        Only mention images in your response if they directly support or illustrate your answer.
        
        Previous conversation context: {chat_history[-3:] if chat_history else 'No previous conversation'}
        
        Question: {query}
        
        Text Context: {context}
        
        {'Available Images for Reference:' if image_descriptions else ''}
        {chr(10).join([f"- {os.path.basename(path)}: {desc['description']}" for path, desc in image_descriptions.items()]) if image_descriptions else ''}
        
        Please provide your response:
        """
        
        content_parts.append(enhanced_prompt)
        
        # Add relevant images inline for context
        for img_path in relevant_images[:5]:  # Limit to 5 most relevant images
            try:
                if os.path.exists(img_path):
                    pil_image = Image.open(img_path)
                    if pil_image.mode == "RGBA":
                        pil_image = pil_image.convert("RGB")
                    content_parts.append(pil_image)
            except Exception as e:
                print(f"Error adding image {img_path} to content: {e}")
        
        # Generate response with images inline
        response = model.generate_content(content_parts)
        return response.text, image_filename_map
        
    except Exception as e:
        print(f"Error generating response with images: {e}")
        # Fallback to text-only response
        return generate_text_only_response(query, context, chat_history), {}

def extract_mentioned_images(response_text, image_filename_map):
    """Extract only the images that are mentioned in the response text"""
    mentioned_images = []
    
    if not response_text or not image_filename_map:
        return mentioned_images
    
    # Convert response to lowercase for case-insensitive matching
    response_lower = response_text.lower()
    
    # Check each image filename against the response text
    for filename, img_path in image_filename_map.items():
        filename_lower = filename.lower()
        
        # Check if the filename (or its variations) is mentioned in the response
        if (filename_lower in response_lower or 
            filename_lower.replace('_', ' ') in response_lower or
            filename_lower.replace('-', ' ') in response_lower or
            os.path.splitext(filename_lower)[0] in response_lower):
            mentioned_images.append(img_path)
            print(f"Found mentioned image: {filename}")
    
    # Also check for common image reference patterns
    image_patterns = [
        r'(?:figure|image|diagram|chart|screenshot|photo|picture)\s*[:\-]?\s*([^\s,\.]+\.(?:png|jpg|jpeg|gif|bmp|svg))',
        r'(?:see|refer to|shown in|displayed in|check)\s+([^\s,\.]+\.(?:png|jpg|jpeg|gif|bmp|svg))',
        r'([^\s,\.]+\.(?:png|jpg|jpeg|gif|bmp|svg))\s+(?:shows|displays|illustrates|demonstrates)'
    ]
    
    for pattern in image_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            # Find the corresponding full path
            for filename, img_path in image_filename_map.items():
                if match.lower() in filename.lower() and img_path not in mentioned_images:
                    mentioned_images.append(img_path)
                    print(f"Found pattern-matched image: {filename}")
    
    print(f"Total mentioned images found: {len(mentioned_images)}")
    return mentioned_images

def generate_text_only_response(query, context, chat_history):
    """Fallback function for text-only response generation"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
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
    
    rag_chain = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt_final
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke({"question": query, "context": context, "chat_history": chat_history})

def rag_pipeline_with_prompt(query, chat_history):
    """Enhanced RAG pipeline that only returns images mentioned in the textual response"""
    
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

    context = "\n".join(documents[0])  # Joining the top documents as context
    
    # Extract potential images from metadata
    potential_images = get_potential_images_from_metadata(metadatas)
    
    # Enhanced image relevance checking with Gemini 2.5 Flash
    relevant_images = []
    image_descriptions = {}
    final_images_to_display = []
    
    if potential_images:
        print(f"Found {len(potential_images)} potential images, performing enhanced relevance analysis with Gemini...")
        relevant_images, image_descriptions = check_image_relevance_with_gemini_enhanced(
            query, potential_images, context
        )
        
        # Generate response that naturally references relevant images
        if relevant_images:
            print(f"Generating response with {len(relevant_images)} relevant images using Gemini 2.5 Flash...")
            try:
                response, image_filename_map = generate_text_response_with_image_references(
                    query, context, relevant_images, image_descriptions, chat_history
                )
                
                # Extract only the images that are mentioned in the response
                final_images_to_display = extract_mentioned_images(response, image_filename_map)
                
            except Exception as e:
                print(f"Error in enhanced response generation: {e}")
                # Fallback to original method
                response = generate_text_only_response(query, context, chat_history)
                final_images_to_display = []
        else:
            print("No relevant images found, generating text-only response...")
            response = generate_text_only_response(query, context, chat_history)
            final_images_to_display = []
    else:
        print("No potential images found, generating text-only response...")
        response = generate_text_only_response(query, context, chat_history)
        final_images_to_display = []
    
    print(f"Context of the query: {context[:200]}...")
    print(f"Final images to display (mentioned in response): {len(final_images_to_display)}")
    
    # Return response and only the images mentioned in the response
    return response, final_images_to_display

# Legacy image path extraction (commented out but kept for reference)
# # Safely extract image path metadata if available
# image_path = None
# if metadatas and isinstance(metadatas[0], list) and metadatas[0] and 'image_file' in metadatas[0][0]:
#     image_meta = metadatas[0][0]
#     file_list_path = image_meta.get('image_file')
#     ref_index = image_meta.get('image_ref_num', 0)
#     try:
#         with open(file_list_path, 'r') as f:
#             image_list = json.load(f)
#         image_path = image_list[ref_index]
#     except Exception as e:
#         print(f"Failed to load image metadata: {e}")
# else:
#     print("No valid image metadata found; skipping image retrieval.")

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
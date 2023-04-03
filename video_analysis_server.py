#!/usr/local/bin/python3

import io
import json
import cv2
import numpy as np
import pinecone
from fastapi import FastAPI, File, UploadFile
from imagehash import phash
from PIL import Image
from typing import List

app = FastAPI()
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_namespace = "image_hashes"

def init_pinecone():
    pinecone.deinit()
    pinecone.init(api_key=pinecone_api_key)

def deinit_pinecone():
    pinecone.deinit()

def get_image_phash(image_data):
    img_array = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return phash(Image.fromarray(img_gray))

@app.post("/upload_image/")
async def upload_image(image: UploadFile = File(...), description: str = ""):
    # Compute the perceptual hash value for the uploaded image
    image_data = await image.read()
    image_phash = get_image_phash(image_data)

    # Prepare the metadata as a JSON string
    metadata = {
        "filename": image.filename,
        "description": description,
    }
    metadata_json = json.dumps(metadata)

    # Initialize Pinecone
    init_pinecone()

    # Create the Pinecone namespace if it doesn't exist
    namespaces = pinecone.list_namespaces()
    if pinecone_namespace not in namespaces:
        pinecone.create_namespace(pinecone_namespace)

    # Create an instance of Pinecone's index
    index = pinecone.Index(index_name=pinecone_namespace)

    # Insert the image hash and metadata into Pinecone
    index.upsert(items={str(image_phash): metadata_json})

    # Deinitialize Pinecone
    deinit_pinecone()

    return {"message": "Image uploaded and indexed successfully"}

@app.post("/search_similar_images/")
async def search_similar_images(image: UploadFile = File(...), top_k: int = 5):
    # Compute the perceptual hash value for the uploaded image
    image_data = await image.read()
    query_phash = get_image_phash(image_data)

    # Initialize Pinecone
    init_pinecone()

    # Create an instance of Pinecone's index
    index = pinecone.Index(index_name=pinecone_namespace)

    # Query Pinecone to find the most similar images
    similar_images = index.fetch(ids=[str(query_phash)], n_results=top_k)

    # Deinitialize Pinecone
    deinit_pinecone()

    # Parse the similar images' metadata
    similar_images_metadata = [
        json.loads(metadata_json) for metadata_json in similar_images[str(query_phash)].values()
    ]

    return {"similar_images": similar_images_metadata}

# Example usage:
# $ uvicorn main:app --reload


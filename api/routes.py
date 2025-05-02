from fastapi import APIRouter, File, UploadFile, HTTPException
from services.face_utils import analyze_faces
# from deepface import DeepFace
# from lightphe import LightPHE
import numpy as np
import os
import tempfile
import insightface
from PIL import Image
from io import BytesIO

router = APIRouter()

# Load model and cache reference embeddings at module level
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

reference_embeddings = {}
reference_names = []

def get_reference_embeddings(folder_path="reference_pictures"):
    global reference_embeddings, reference_names
    if reference_embeddings:
        return reference_embeddings, reference_names
    reference_embeddings = {}
    reference_names = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img = np.array(Image.open(img_path).convert("RGB"))
            faces = model.get(img)
            if faces:
                emb = faces[0].embedding
                reference_embeddings[file] = emb
                reference_names.append(file)
    return reference_embeddings, reference_names

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@router.post(
    "/compare-faces/",
    summary="Compare Two Face Images",
    description="Upload two images. The API will compare the faces and return the similarity analysis and confidence."
)
async def compare_faces(
    image1: UploadFile = File(..., description="First image file"),
    image2: UploadFile = File(..., description="Second image file")
):
    """
    Compare two face images and return analysis results with confidence.
    """
    img1_bytes = await image1.read()
    img2_bytes = await image2.read()
    analysis_result = await analyze_faces(img1_bytes, img2_bytes)
    distance = analysis_result.get('distance', None)
    threshold = analysis_result.get('threshold', None)
    if distance is not None and threshold is not None:
        confidence = 1 - min(1, distance / threshold)
        confidence_percent = round(confidence * 100, 2)
    else:
        confidence_percent = None
    return {
        "analysis": analysis_result,
        "confidence_percent": confidence_percent,
    }

@router.post(
    "/compare-with-employees/",
    summary="Compare Image With Employees Folder (InsightFace)",
    description="Upload an image. The API will compare it with all images in the reference_pictures folder using InsightFace embeddings."
)
async def compare_with_employees(
    image: UploadFile = File(..., description="Image file to compare (e.g., from your camera)")
):
    folder_path = "reference_pictures"
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=500, detail="Reference folder does not exist.")

    # Read uploaded image
    img_bytes = await image.read()
    img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))

    # Get embedding for uploaded image
    faces = model.get(img)
    if not faces:
        return {"match": False, "employee": "Unknown", "similarity": None}
    query_emb = faces[0].embedding

    # Get reference embeddings
    ref_embs, ref_names = get_reference_embeddings(folder_path)

    # Compare
    best_similarity = -1
    best_file = None
    for name, emb in ref_embs.items():
        sim = cosine_similarity(query_emb, emb)
        if sim > best_similarity:
            best_similarity = sim
            best_file = name

    threshold = 0.5  # You may want to tune this threshold
    if best_similarity > threshold:
        return {
            "match": True,
            "employee": best_file,
            "similarity": float(best_similarity)
        }
    else:
        return {
            "match": False,
            "employee": "Unknown",
            "similarity": float(best_similarity)
        } 
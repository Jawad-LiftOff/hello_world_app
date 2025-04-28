from fastapi import APIRouter, File, UploadFile, HTTPException
from services.face_utils import analyze_faces
from deepface import DeepFace
import os
import tempfile

router = APIRouter()

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
    summary="Compare Image With Employees Folder (Fast)",
    description="Upload an image. The API will compare it with all images in the reference_pictures folder using DeepFace's fast search."
)
async def compare_with_employees(
    image: UploadFile = File(..., description="Image file to compare (e.g., from your camera)")
):
    """
    Compare an uploaded image with all images in the reference_pictures folder using DeepFace.find.
    Returns the best match (if any) with confidence and the matched file name.
    """
    folder_path = "reference_pictures"
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=500, detail="Reference folder does not exist.")

    # Save uploaded image to a temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        results = DeepFace.find(
            img_path=tmp_path,
            db_path=folder_path,
            model_name="ArcFace",  # or your preferred model
            enforce_detection=True
        )
        # results is a list of DataFrames, one per model (if multiple models used)
        # We'll use the first DataFrame
        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            return {
                "match": True,
                "employee": os.path.basename(best_match['identity']),
                "distance": float(best_match['distance']),
                "confidence_percent": round((1 - min(1, best_match['distance'] / best_match['threshold'])) * 100, 2)
            }
        else:
            return {"match": False, "employee": "Unknown", "confidence_percent": None}
    finally:
        os.remove(tmp_path) 
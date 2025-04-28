from fastapi import APIRouter, File, UploadFile, HTTPException
from services.face_utils import analyze_faces
import os

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
    summary="Compare Image With Employees Folder",
    description="Upload an image (e.g., from your camera). The API will compare it with all images in the reference_pictures folder and return the best match, including the file name."
)
async def compare_with_employees(
    image: UploadFile = File(..., description="Image file to compare (e.g., from your camera)")
):
    """
    Compare an uploaded image with all images in the reference_pictures folder.
    Returns the best match (if any) with confidence and the matched file name.
    """
    img_bytes = await image.read()
    folder_path = "reference_pictures"
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=500, detail="Reference folder does not exist.")
    reference_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not reference_files:
        raise HTTPException(status_code=404, detail="No reference images found.")
    best_result = None
    best_file = None
    best_confidence = -1
    for ref_file in reference_files:
        ref_path = os.path.join(folder_path, ref_file)
        with open(ref_path, "rb") as f:
            ref_bytes = f.read()
        try:
            result = await analyze_faces(img_bytes, ref_bytes)
            distance = result.get('distance', None)
            threshold = result.get('threshold', None)
            if distance is not None and threshold is not None:
                confidence = 1 - min(1, distance / threshold)
            else:
                confidence = 0
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = result
                best_file = ref_file
        except Exception:
            continue
    if best_result is None:
        return {"match": False, "message": "No matches found."}
    return {
        "match": best_result.get('verified', False),
        "reference_file": best_file,
        "confidence_percent": round(best_confidence * 100, 2),
        "analysis": best_result
    } 
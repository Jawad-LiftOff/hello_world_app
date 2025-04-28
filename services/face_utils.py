import os
from fastapi import HTTPException
from deepface import DeepFace
from PIL import Image
import io

async def analyze_faces(img1_bytes: bytes, img2_bytes: bytes):
    """Analyze and compare two face images using DeepFace."""
    try:
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))
        temp_path1 = "temp_img1.jpg"
        temp_path2 = "temp_img2.jpg"
        img1.save(temp_path1)
        img2.save(temp_path2)
        result = DeepFace.verify(
            img1_path=temp_path1,
            img2_path=temp_path2,
            model_name="ArcFace",
            enforce_detection=True
        )
        os.remove(temp_path1)
        os.remove(temp_path2)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 
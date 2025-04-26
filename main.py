import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import numpy as np
from PIL import Image
import io
import openai
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# Initialize OpenAI client
print(os.getenv('OPENAI_API_KEY'), 'API KEY!!!')
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = FastAPI(title="Image Comparison API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def analyze_faces(img1_bytes: bytes, img2_bytes: bytes):
    """Analyze and compare two face images using DeepFace."""
    try:
        # Convert bytes to PIL Images
        img1 = Image.open(io.BytesIO(img1_bytes))
        img2 = Image.open(io.BytesIO(img2_bytes))
        
        # Save temporary files for DeepFace
        temp_path1 = "temp_img1.jpg"
        temp_path2 = "temp_img2.jpg"
        img1.save(temp_path1)
        img2.save(temp_path2)

        # Analyze faces using DeepFace
        result = DeepFace.verify(
            img1_path=temp_path1,
            img2_path=temp_path2,
            model_name="VGG-Face",
            enforce_detection=False
        )

        # Clean up temporary files
        os.remove(temp_path1)
        os.remove(temp_path2)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def generate_comparison_explanation(analysis_result: dict) -> str:
    """Generate natural language explanation using GPT-4."""
    try:
        similarity = analysis_result.get('distance', 0)
        verified = analysis_result.get('verified', False)
        
        prompt = f"""
        Analyze these face comparison results and provide a natural language explanation:
        - Similarity score: {similarity}
        - Match verified: {verified}
        
        Please describe what these results mean in simple terms and what conclusions we can draw 
        about the facial comparison.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in facial recognition analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

@app.post("/compare-faces/")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """
    Compare two face images and return analysis results with explanation.
    """
    try:
        # Read image files
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        # Analyze faces
        analysis_result = await analyze_faces(img1_bytes, img2_bytes)
        
        # Generate explanation
        explanation = await generate_comparison_explanation(analysis_result)

        return {
            "analysis": analysis_result,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
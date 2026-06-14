from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="ShineTicket AI Service")

# Load model (download on first run)
MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading model {MODEL_NAME}...")
try:
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]

@app.get("/")
def read_root():
    return {"status": "AI Service is running", "model": MODEL_NAME}

@app.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(req: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    if not req.text or req.text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    try:
        # Generate embedding
        # encode returns numpy array, we convert to list of floats
        embedding = model.encode(req.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Chạy ở port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

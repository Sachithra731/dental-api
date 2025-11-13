# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import sys
import os

app = FastAPI(title="Dental Disease Detection API")

# CORS - Allow all origins (since web app will be on GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

CLASSES = ["CALCULUS", "CARIES", "GINGIVITIS", "HYPODONTIA", "TOOTH DISCOLORATION", "ULCERS"]
NUM_CLASSES = len(CLASSES)
THRESHOLD = 70.0  # percent

# Image transforms
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_checkpoint(path):
    """Load model checkpoint with robust error handling"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    ckpt = torch.load(path, map_location="cpu")
    
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint does not contain a state dict.")

    # Strip 'module.' prefix if present
    new_state = {}
    for k, v in ckpt.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_state[new_k] = v
    return new_state

# Build model
print("[INFO] Loading model...")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load weights
try:
    model_path = "oral_disease_model.pth"
    sd = load_checkpoint(model_path)
    
    # Check if fc layer matches
    if "fc.weight" in sd and sd["fc.weight"].shape[0] != NUM_CLASSES:
        print(f"[WARNING] Model has {sd['fc.weight'].shape[0]} classes, expected {NUM_CLASSES}")
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd)
    
    print(f"[INFO] Model loaded successfully from {model_path}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
    raise

model.eval()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Dental Disease Detection API",
        "status": "running",
        "model": "ResNet50",
        "classes": CLASSES,
        "version": "1.0"
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict dental disease from uploaded image
    
    Returns:
        JSON with label, confidence, top predictions, and all probabilities
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Transform and predict
    x = tfm(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=min(3, NUM_CLASSES))

    top_indices = topk.indices.tolist()
    top_probs = topk.values.tolist()

    # Build top predictions list
    top_list = []
    for idx, p in zip(top_indices, top_probs):
        label = CLASSES[idx] if 0 <= idx < NUM_CLASSES else f"CLASS_{idx}"
        top_list.append({
            "label": label,
            "confidence": round(p * 100, 2),
            "index": idx
        })

    highest_conf = top_probs[0] * 100
    highest_label = CLASSES[top_indices[0]]

    # Apply threshold
    if highest_conf < THRESHOLD:
        out_label = "No disease"
        out_conf = round(highest_conf, 2)
    else:
        out_label = highest_label
        out_conf = round(highest_conf, 2)

    # Full probability map
    probs_map = {
        CLASSES[i]: round(float(probs[i]) * 100, 2) 
        for i in range(NUM_CLASSES)
    }

    # Log prediction
    print(f"[PREDICT] {out_label} ({out_conf}%) - Top: {top_list[0]['label']}")

    return JSONResponse({
        "label": out_label,
        "confidence": out_conf,
        "top": top_list,
        "probs": probs_map,
        "threshold": THRESHOLD
    })

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all error handler"""
    print(f"[ERROR] {type(exc).__name__}: {str(exc)}", file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
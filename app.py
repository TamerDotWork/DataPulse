import io
import os
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "server_model.pkl"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the Training UI (Step 1)"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request):
    """Serve the Prediction UI (Step 2)"""
    return templates.TemplateResponse("predict.html", {"request": request})

# ==========================
#  API: AUTO-TRAIN
# ==========================
@app.post("/api/train")
async def train_model(
    file: UploadFile = File(...)
):
    try:
        # 1. Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # 2. Select Numerical Columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.empty:
            raise HTTPException(status_code=400, detail="No numerical columns found in dataset.")

        features = numeric_df.columns.tolist()

        # 3. Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # 4. AUTO-DETECT BEST K (Silhouette Score)
        # We test K from 2 up to 10 (or len(df) if small)
        max_k = min(10, len(numeric_df))
        if max_k < 2:
            raise HTTPException(status_code=400, detail="Not enough data points to cluster.")

        best_score = -1
        best_k = 2
        best_model = None

        # Loop to find optimal K
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Calculate score (higher is better)
            score = silhouette_score(X_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans

        # 5. Create Artifact with the BEST model
        model_artifact = {
            'model': best_model,
            'scaler': scaler,
            'features': features,
            'k_value': best_k,
            'score': best_score
        }

        # 6. Save to Disk
        joblib.dump(model_artifact, MODEL_PATH)

        return JSONResponse(content={
            "message": f"Training successful! Optimal clusters found: {best_k} (Score: {best_score:.2f})", 
            "redirect": "/result",
            "k_found": best_k
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================
#  API: PREDICT
# ==========================
@app.post("/api/predict")
async def predict(
    data_file: UploadFile = File(...)
):
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail="Model not found. Please train the model first.")

        # Load Model
        artifact = joblib.load(MODEL_PATH)
        model = artifact['model']
        scaler = artifact['scaler']
        features = artifact['features']

        # Load Data
        data_content = await data_file.read()
        df = pd.read_csv(io.BytesIO(data_content))

        # Validation
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Predict
        X = df[features]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        df['Cluster_ID'] = predictions

        # Export
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response_buffer = io.BytesIO(stream.getvalue().encode())

        return StreamingResponse(
            response_buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
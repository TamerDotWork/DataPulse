import io
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = FastAPI()


templates = Jinja2Templates(directory="templates")

# Mount static folder if you have one (optional, but good practice)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the HTML UI"""
    return templates.TemplateResponse("index.html", {"request": request})

# ==========================
#  API: TRAIN MODEL
# ==========================
@app.post("/api/train")
async def train_model(
    file: UploadFile = File(...),
    n_clusters: int = Form(...)
):
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Select Numerical Columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.empty:
            raise HTTPException(status_code=400, detail="No numerical columns found in dataset.")

        features = numeric_df.columns.tolist()

        # Training Pipeline
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_scaled)

        # Create Artifact
        model_artifact = {
            'model': kmeans,
            'scaler': scaler,
            'features': features
        }

        # Serialize
        buffer = io.BytesIO()
        joblib.dump(model_artifact, buffer)
        buffer.seek(0)

        # Return File
        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=model.pkl"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================
#  API: PREDICT
# ==========================
@app.post("/api/predict")
async def predict(
    data_file: UploadFile = File(...),
    model_file: UploadFile = File(...)
):
    try:
        # Load Model
        model_content = await model_file.read()
        artifact = joblib.load(io.BytesIO(model_content))
        
        model = artifact['model']
        scaler = artifact['scaler']
        features = artifact['features']

        # Load New Data
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

        # Append Result
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
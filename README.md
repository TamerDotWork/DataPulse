<div align="center">
  <img src="https://github.com/TamerDotWork/datapulse/blob/main/cover.jpg" alt="DataPulse Demo" width="auto" style="border-radius: 16px;">
</div>

<div align="center">

# DataPulse

### Automated clustering & pattern discovery using FastAPI, Pandas, and Scikit-Learn

<br/>

</div>

<br/>

<div align="center">
  <img src="https://github.com/TamerDotWork/datapulse/blob/main/screenshot.jpg" alt="DataPulse UI" width="auto" style="border-radius: 16px;">
</div>

<br>

> [!TIP]
> DataPulse automatically discovers hidden patterns in your datasets by selecting the optimal number of clusters — no tuning required.

---

## DataPulse Overview

**DataPulse** is an automated data intelligence service that transforms raw tabular datasets into **clustered, insight-ready data** through a simple two-step workflow.

Built with:

- FastAPI backend  
- Pandas data processing  
- Scikit-Learn clustering  
- Automatic preprocessing & scaling  
- Persisted ML artifacts  

**Execution flow:**

**upload → preprocess → auto-train → evaluate → cluster → export**

---

## Key Capabilities

- Automatic cluster detection (Silhouette Score)
- Zero-configuration KMeans training
- Smart numeric feature selection
- Reusable trained models
- CSV export with `Cluster_ID`
- Clean web interface (Train → Predict)

---

## Use Cases

- Customer & user segmentation  
- Pattern discovery in datasets  
- Exploratory data analysis  
- Pre-ML data structuring  
- BI & reporting preparation  

---

## Quick Start

### Prerequisites

- Python 3.10+  
- pip / virtual environment  

---

### Installation

```bash
# Clone repository
git clone https://github.com/TamerDotWork/datapulse
cd datapulse

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run DataPulse
python app.py

```
<p align="center">
  <a href="https://tamer.work">
    <img src="https://github.com/TamerDotWork/datapulse/blob/main/logo.jpg" alt="Vesper Banner" width="auto" style="border-radius: 56px;">
  </a>
</p>
# 🌱 PlantGuard AI — Plant Disease Prediction & Pesticide Recommendation System

A full-stack AI application that detects plant diseases from leaf images and recommends pesticide treatments. Built with a custom CNN model, FastAPI backend, and React frontend.

---

## 📸 Features

- **Upload or capture** plant leaf photos (desktop + mobile camera)
- **CNN model** trained on 26 disease classes from the PlantVillage dataset
- **Instant diagnosis**: disease name, confidence score, and severity
- **Treatment recommendations**: pesticide name, dosage, and precautions
- **Low confidence detection**: prompts user for a better image if confidence < 70%
- **Top-5 predictions** for transparency
- **Responsive design**: works on mobile and desktop

---

## 🗂 Project Structure

```
plant-disease-system/
├── backend/                    # FastAPI backend
│   ├── main.py                 # API routes and app entry point
│   ├── requirements.txt        # Python dependencies
│   └── utils/
│       ├── model_loader.py     # Model loading and inference
│       └── disease_info.py     # Knowledge base loader
│
├── frontend/                   # React.js frontend
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js              # All pages and components
│   │   ├── App.css             # Styling
│   │   └── index.js            # Entry point
│   └── package.json
│
├── model/
│   ├── train.py                # CNN training script
│   ├── requirements.txt        # Training dependencies
│   ├── model.h5                # Trained model (generated after training)
│   └── class_indices.json      # Class label mapping (generated after training)
│
├── data/
│   ├── disease_info.json       # Disease knowledge base (26 entries)
│   └── prepare_data.py         # Dataset download + train/val split script
│
├── render.yaml                 # Render.com deployment config
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ 
- Node.js 18+
- Kaggle account (for dataset download)

---

### Step 1 — Clone & Setup

```bash
git clone <your-repo-url>
cd plant-disease-system
```

---

### Step 2 — Train the Model

#### 2a. Install training dependencies
```bash
cd model
pip install -r requirements.txt
cd ..
```

#### 2b. Configure Kaggle API
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 2c. Download and prepare dataset
```bash
python data/prepare_data.py
```
This will:
- Download the PlantVillage dataset from Kaggle
- Split into `data/train/` and `data/val/` (80/20 split)

#### 2d. Train the CNN model
```bash
python model/train.py
```
This will:
- Train the CNN with early stopping
- Save `model/model.h5` and `model/class_indices.json`
- Save a training history plot to `model/training_history.png`

⏱ Training time: ~30–60 minutes on GPU, 2–4 hours on CPU.

**To use GPU acceleration:**
```bash
pip install tensorflow[and-cuda]  # For NVIDIA GPU
```

---

### Step 3 — Run the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: `http://localhost:8000`

Test the API:
```bash
curl http://localhost:8000/health
```

---

### Step 4 — Run the Frontend

```bash
cd frontend
npm install
npm start
```

Frontend will open at: `http://localhost:3000`

---

## 🔌 API Reference

### `GET /health`
Returns API status and model loading state.

### `POST /predict`
Upload a plant leaf image for disease prediction.

**Request:**
```
Content-Type: multipart/form-data
file: <image.jpg>
```

**Response (healthy plant):**
```json
{
  "disease": "Healthy Tomato",
  "confidence": 97.3,
  "low_confidence": false,
  "solution": {
    "disease_name": "Healthy Tomato",
    "plant": "Tomato",
    "symptoms": "No disease present...",
    "pesticide": "No pesticide required",
    "precautions": ["Regular scouting..."],
    "severity": "None"
  },
  "top5": { ... }
}
```

**Response (low confidence):**
```json
{
  "low_confidence": true,
  "confidence": 45.2,
  "message": "Confidence too low. Please try a clearer image...",
  "top5": { ... }
}
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
Conv2D(32) → BatchNorm → MaxPool
    ↓
Conv2D(64) → BatchNorm → MaxPool
    ↓
Conv2D(128) → BatchNorm → MaxPool
    ↓
Conv2D(256) → BatchNorm → MaxPool
    ↓
Flatten
    ↓
Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.3)
    ↓
Dense(26, softmax)
```

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Early Stopping**: patience=5 on val_accuracy
- **Expected Accuracy**: 90–96% on validation set

---

## 🌿 Supported Plant Diseases (26 Classes)

| Plant | Diseases |
|-------|----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Orange | Citrus Greening (HLB) |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🚢 Deployment

### Backend → Render.com (Free Tier)

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repository
4. Set **Build Command**: `cd backend && pip install -r requirements.txt`
5. Set **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `PYTHON_VERSION = 3.10.0`
7. Deploy!

> **Important**: Upload `model.h5` and `class_indices.json` to your repo or a cloud storage bucket (AWS S3, GCS) and update paths in `main.py` accordingly. The free Render tier has a 1GB disk limit.

### Frontend → Netlify or Vercel

#### Vercel (recommended)
```bash
cd frontend
npm install -g vercel
vercel
```

#### Netlify
```bash
cd frontend
npm run build
# Drag & drop the build/ folder to netlify.com/drop
```

#### Set environment variable in Vercel/Netlify:
```
REACT_APP_API_URL=https://your-render-backend-url.onrender.com
```

---

## ⚙️ Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Confidence threshold | `backend/utils/model_loader.py` | 0.70 (70%) |
| Image input size | `model/train.py` + `model_loader.py` | 224×224 |
| API port | `uvicorn` command | 8000 |
| Frontend API URL | `.env` → `REACT_APP_API_URL` | `http://localhost:8000` |

---

## 🐛 Troubleshooting

**"Model file not found"**  
→ Make sure you've run `python model/train.py` and `model/model.h5` exists.

**CORS errors in browser**  
→ Ensure the backend is running and `REACT_APP_API_URL` matches the backend URL.

**Camera not working on mobile**  
→ Ensure you're serving the frontend over HTTPS (required for camera access on mobile).

**Low accuracy predictions**  
→ Make sure the leaf fills most of the frame. Use good lighting. Avoid blurry images.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with TensorFlow, FastAPI, and React*

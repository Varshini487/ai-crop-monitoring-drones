# 🚁 AI Crop Monitoring with Drones

A **computer vision pipeline** that processes aerial drone imagery to detect crop stress, identify diseases, and estimate yield — turning raw drone footage into actionable farm intelligence.

## 🧠 How It Works (4-step pipeline)

1. **Image Ingestion** — uploads or streams drone photos (RGB/multispectral)
2. **Preprocessing** — orthorectification (aligns to GPS), normalization, tiling (divide large mosaic into 512×512 cells)
3. **Analysis Models** — runs 3 parallel inference engines:
   - **Stress Detection** — NDVI (Normalized Difference Vegetation Index) calculates greenness; low NDVI → crop stress
   - **Disease Classifier** — fine-tuned ResNet50 flags diseased patches (blight, rust, etc.)
   - **Yield Estimator** — XGBoost predicts yield (kg/ha) from vegetation density + field history
4. **Output** — heatmaps show stress zones + disease locations; farmer gets report: "Field B: 15% diseased area (southeast corner), estimated 8.2 T/ha"

## 🛠️ Tech Stack
- **OpenDroneMap / Pix4D** – drone image stitching & geo-registration (or use demo tiles)
- **GDAL/Rasterio** – raster processing
- **TensorFlow/PyTorch** – CNN models (ResNet, EfficientNet)
- **Scikit-learn** – NDVI + yield regression
- **Folium** – interactive map visualization
- **Streamlit** – web interface

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/ai-crop-monitoring-drones
cd ai-crop-monitoring-drones
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Use Cases
- Early disease/stress detection → spray only affected areas (reduce chemicals)
- Yield prediction → harvest planning, market forecasting
- Insurance claims → objective damage assessment
- Precision agriculture optimization

## 🎤 Interview Talking Points

**1️⃣ Multispectral imaging beats RGB for agricultural insight.**
"Drones with just RGB cameras show what your eye sees. But multispectral drones capture infrared + red-edge bands invisible to humans. NDVI (greenness index) uses these: (NIR - Red) / (NIR + Red). Healthy plants reflect tons of infrared; stressed plants don't. NDVI alone detects stress 2-3 weeks *before* visible symptoms. Early detection = way more effective treatment."

**2️⃣ Georeferencing turns imagery into actionable maps.**
"Raw drone photos are pretty but useless for farming. Orthomosaicing aligns every pixel to GPS coordinates. Then you overlay your field boundary, divide into management zones, and report: 'Southeast 2 hectares have 20% disease.' Farmer can pinpoint exactly where to spray. Without geo-registration, it's just eye candy."

**3️⃣ Multi-model ensemble beats any single model.**
"One model might miss subtle rust; another might flag healthy plants as diseased. Running NDVI *and* a CNN classifier *and* historical yield correlation gives you confidence. You can say 'this area is 92% likely diseased' (high confidence, high precision) vs. one model saying 'maybe diseased.' Ensemble + confidence thresholds = tool farmers actually trust."


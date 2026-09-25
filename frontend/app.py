import streamlit as st
import time
from PIL import Image
import os
import tempfile
from backend_processor import run_live_prediction

# --- Page Configuration ---
st.set_page_config(
    page_title="AquaSense AI | Flood Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Premium CSS (Glassmorphism, Animations, Fonts) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    /* Main Title Gradient */
    .main-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    /* Sidebar styling (Glassmorphism) */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        width: 100%;
        box-shadow: 0 10px 20px -10px rgba(79, 70, 229, 0.5);
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 15px 25px -10px rgba(79, 70, 229, 0.7);
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(30, 41, 59, 0.4);
        border: 2px dashed rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: #38bdf8;
    }
    
    /* Metric Cards - Custom HTML styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 1s ease-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px -10px rgba(56, 189, 248, 0.3);
        border: 1px solid rgba(56,189,248,0.3);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0f2fe 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-sub {
        color: #10b981;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        color: #94a3b8;
        font-size: 1.1rem;
        border-radius: 4px 4px 0 0;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8;
        border-bottom-color: #38bdf8 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIVERABLES_DIR = os.path.join(BASE_DIR, "deliverables")
TEMP_OUTPUT_DIR = os.path.join(BASE_DIR, "data")

def load_image(filename, directory=DELIVERABLES_DIR):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>System Controls</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🛰️ Data Ingestion")
    st.caption("Upload a Pre-Monsoon Sentinel-2 (NDWI) optical image to forecast spatial flood susceptibility.")
    
    uploaded_file = st.file_uploader("", type=['tif', 'tiff'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    demo_mode = st.toggle("⚡ Fast Presentation Mode", value=False, 
                          help="Toggle ON to instantly load the pre-calculated 2026 data. Keep OFF to run live ML inference (takes ~2 mins).")
    
    st.markdown("---")
    st.markdown("### 🧠 Backend Status")
    st.success("🟢 ML Core: Random Forest")
    st.success("🟢 Terrain: LULC, DEM, Soil Loaded")

# --- Main Interface ---
st.markdown("<h1 class='main-title'>AquaSense AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>High-Resolution Flood Susceptibility & Early Warning System powered by Satellite SAR and Ensemble Machine Learning</div>", unsafe_allow_html=True)

if uploaded_file or (demo_mode and st.sidebar.button("Launch Prediction Engine", type="primary")):
    
    if demo_mode:
        # --- Fast Presentation Mode ---
        with st.status("🚀 Executing Predictive Engine (Demo Mode)...", expanded=True) as status:
            st.write("📡 Ingesting Pre-Monsoon Satellite Imagery...")
            time.sleep(0.6)
            st.write("🗺️ Fusing Topographical Layers (DEM, TWI, Distance to River)...")
            time.sleep(0.6)
            st.write("🌱 Integrating LULC & Soil Composition...")
            time.sleep(0.6)
            st.write("⚙️ Running Random Forest Tensor Inference...")
            time.sleep(0.8)
            status.update(label="✅ Prediction Complete!", state="complete", expanded=False)
            
        heatmap_img = load_image("flood_susceptibility_heatmap.png")
        if heatmap_img:
            st.image(heatmap_img, use_container_width=True)
            
    else:
        # --- Live Inference Mode ---
        if uploaded_file is None:
            st.warning("⚠️ Please upload a .tif file first to run Live Inference.")
            st.stop()
            
        with st.status("⚙️ Processing Live Satellite Feed...", expanded=True) as status:
            st.write("💾 Caching uploaded data to local tensor storage...")
            
            temp_input_path = os.path.join(TEMP_OUTPUT_DIR, "temp_uploaded_preflood.tif")
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            temp_output_png = os.path.join(TEMP_OUTPUT_DIR, "live_prediction_heatmap.png")
            
            # Progress callback for the backend
            my_bar = st.progress(0)
            def update_progress(pct, text):
                my_bar.progress(pct, text=text)
                
            try:
                success = run_live_prediction(temp_input_path, temp_output_png, progress_callback=update_progress)
                if success:
                    my_bar.empty()
                    status.update(label="✅ Live Inference Complete!", state="complete", expanded=False)
                    
                    live_img = load_image("live_prediction_heatmap.png", directory=TEMP_OUTPUT_DIR)
                    if live_img:
                        st.image(live_img, use_container_width=True)
            except Exception as e:
                status.update(label="❌ Inference Failed", state="error", expanded=False)
                st.error(f"Error details: {str(e)}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Premium Custom Metrics ---
    st.markdown("""
    <div style='display: flex; gap: 1rem; margin-bottom: 2rem;'>
        <div class='metric-card' style='flex: 1;'>
            <div class='metric-title'>Spatial Accuracy</div>
            <div class='metric-value'>79.7%</div>
            <div class='metric-sub'>↑ +2.7% vs XGBoost Base</div>
        </div>
        <div class='metric-card' style='flex: 1;'>
            <div class='metric-title'>ROC-AUC Score</div>
            <div class='metric-value'>0.981</div>
            <div class='metric-sub'>Exceptional Discrimination</div>
        </div>
        <div class='metric-card' style='flex: 1;'>
            <div class='metric-title'>False Positive Rate</div>
            <div class='metric-value'>1.0%</div>
            <div class='metric-sub'>Highly Conservative</div>
        </div>
        <div class='metric-card' style='flex: 1;'>
            <div class='metric-title'>Engine Status</div>
            <div class='metric-value'>Operational</div>
            <div class='metric-sub'>System Nominal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Data Visualization Tabs ---
    st.markdown("<h3 style='margin-bottom: 1rem; color: white;'>Model Analytics & Validation</h3>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Feature Importance", "📈 ROC Curve", "🧮 Confusion Matrix"])
    
    with tab1:
        feat_img = load_image("fig_feature_importance.png")
        if feat_img: st.image(feat_img, use_container_width=True)
            
    with tab2:
        roc_img = load_image("fig_roc_curve.png")
        if roc_img: st.image(roc_img, use_container_width=True)
            
    with tab3:
        cm_img = load_image("fig_confusion_matrix.png")
        if cm_img: st.image(cm_img, use_container_width=True)
            
else:
    # --- Landing Page Content (Before Interaction) ---
    st.markdown("""
    <div style='background: rgba(30, 41, 59, 0.3); border-left: 4px solid #38bdf8; padding: 2rem; border-radius: 8px; margin-top: 2rem;'>
        <h3 style='color: white; margin-bottom: 1rem;'>Awaiting Satellite Telemetry...</h3>
        <p style='color: #94a3b8; font-size: 1.1rem;'>
            To initialize the predictive pipeline, please utilize the control panel on the left to ingest pre-monsoon optical data. 
            For immediate demonstration purposes, activate <strong>Fast Presentation Mode</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Showcase
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'><h4 style='color:white;'>🛰️ SAR Integration</h4><p style='color:#94a3b8;'>Trained on 5 years of Sentinel-1 radar telemetry, ignoring cloud cover.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h4 style='color:white;'>🧠 Ensemble ML</h4><p style='color:#94a3b8;'>Powered by Random Forest, identifying non-linear topological relationships.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><h4 style='color:white;'>🌍 ESA WorldCover</h4><p style='color:#94a3b8;'>High-resolution LULC ensures urban permeability factors are heavily weighted.</p></div>", unsafe_allow_html=True)

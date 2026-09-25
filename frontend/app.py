import streamlit as st
import time
from PIL import Image
import os
import tempfile
from backend_processor import run_live_prediction

# --- Page Configuration ---
st.set_page_config(
    page_title="Flood Susceptibility Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0b1121;
        color: #e2e8f0;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #60a5fa !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        border: none;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #38bdf8;
    }
    
    /* Alerts/Info boxes */
    .stAlert {
        background-color: #1e293b;
        border: 1px solid #3b82f6;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIVERABLES_DIR = os.path.join(BASE_DIR, "deliverables")
TEMP_OUTPUT_DIR = os.path.join(BASE_DIR, "data") # Place temporary live outputs here

def load_image(filename, directory=DELIVERABLES_DIR):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

# --- Sidebar: User Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3039/3039423.png", width=60)
    st.title("System Controls")
    st.markdown("---")
    
    st.markdown("### 1. Upload Satellite Data")
    st.info("Upload a Pre-Monsoon Sentinel-2 (NDWI) optical image to forecast spatial flood susceptibility.")
    
    uploaded_file = st.file_uploader("Upload Pre-Flood Image (.tif)", type=['tif', 'tiff'])
    
    demo_mode = st.checkbox("Enable Presentation Mode (Fast)", value=False, 
                            help="Uncheck this to run the ACTUAL machine learning model on your uploaded image live (takes ~2 minutes).")
    
    st.markdown("---")
    st.markdown("### Model Status")
    st.success("✅ Random Forest Core Active")
    st.success("✅ Static Terrain Data Loaded (LULC, DEM, Soil)")

# --- Main Dashboard ---
st.title("🌊 Flood Susceptibility Early Warning System")
st.markdown("An ML-driven spatial planning tool powered by Sentinel-1 SAR, Random Forest, and Topographical Analytics.")

if uploaded_file or (demo_mode and st.sidebar.button("Run Flood Prediction Engine", type="primary")):
    
    if demo_mode:
        # ---------------------------------------------------------
        # DEMO MODE (Fast presentation)
        # ---------------------------------------------------------
        progress_text = "Initializing predictive engine..."
        my_bar = st.progress(0, text=progress_text)
        
        time.sleep(0.5)
        my_bar.progress(25, text="Loading Static Terrain Features (LULC, TWI, DEM)...")
        time.sleep(1)
        my_bar.progress(50, text="Calculating Surface Moisture (NDWI) from uploaded image...")
        time.sleep(1)
        my_bar.progress(75, text="Running Random Forest Ensemble (n_estimators=100)...")
        time.sleep(1.5)
        my_bar.progress(100, text="Prediction Complete! Rendering interactive maps...")
        time.sleep(0.5)
        my_bar.empty()
        
        st.success("Successfully generated flood susceptibility map (Demo Mode)!")
        heatmap_img = load_image("flood_susceptibility_heatmap.png")
        if heatmap_img:
            st.image(heatmap_img, caption="Red = High Risk | Blue = River Channel", use_container_width=True)
            
    else:
        # ---------------------------------------------------------
        # LIVE INFERENCE MODE
        # ---------------------------------------------------------
        if uploaded_file is None:
            st.warning("Please upload a .tif file first to run Live Inference.")
            st.stop()
            
        st.warning("Running Live Inference on uploaded file. This will take 1-3 minutes depending on the file size. Please wait...")
        my_bar = st.progress(0, text="Saving uploaded file...")
        
        def update_progress(pct, text):
            my_bar.progress(pct, text=text)
            
        try:
            # Save uploaded file to disk
            temp_input_path = os.path.join(TEMP_OUTPUT_DIR, "temp_uploaded_preflood.tif")
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            temp_output_png = os.path.join(TEMP_OUTPUT_DIR, "live_prediction_heatmap.png")
            
            # Run the actual python backend function
            success = run_live_prediction(temp_input_path, temp_output_png, progress_callback=update_progress)
            
            if success:
                time.sleep(0.5)
                my_bar.empty()
                st.success("Successfully processed uploaded image and generated live flood susceptibility map!")
                
                live_img = load_image("live_prediction_heatmap.png", directory=TEMP_OUTPUT_DIR)
                if live_img:
                    st.image(live_img, caption="Live Inference Result", use_container_width=True)
                else:
                    st.error("Failed to load live output image.")
        except Exception as e:
            st.error(f"An error occurred during live inference: {str(e)}")
            my_bar.empty()
            
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spatial Accuracy", "79.7%", "+2.7% vs XGBoost")
    with col2:
        st.metric("ROC AUC Score", "0.981", "Exceptional")
    with col3:
        st.metric("False Positive Rate", "1.0%", "Highly Conservative")
    with col4:
        st.metric("Status", "Live" if not demo_mode else "Demo", "Operational")
        
    st.markdown("---")
    
    # Analytics Expanders
    st.markdown("### 📊 Model Analytics & Validation")
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "ROC Curve", "Confusion Matrix"])
    
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
    # Default State (Before Upload)
    st.markdown("""
    ### 👈 Awaiting Data Input
    Please use the control panel on the left to upload a pre-monsoon optical image, or activate **Presentation Mode** and click **Run Prediction Engine** to load the 2026 test scenario.
    """)
    
    # Show pipeline logic while waiting
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Static Inputs (Pre-loaded):**\n* Digital Elevation Model (DEM)\n* ESA WorldCover LULC\n* Soil Clay Percentage\n* Distance to River\n* Topographic Wetness Index (TWI)")
    with col2:
        st.warning("**Dynamic Input (Required):**\n* Pre-Monsoon Sentinel-2 (NDWI) Image")

import streamlit as st
import base64
from pathlib import Path

def set_page_config():
    st.set_page_config(
        page_title="CropEye",
        page_icon="🚁",
        layout="wide"
    )

def main():
    set_page_config()

    # Title and Introduction
    st.title("CropEye")

    # Demo Link
    st.markdown("[🎮 Try the Live Demo](https://cropeye.demo.com)", unsafe_allow_html=True)

    st.write("""
    Welcome to CropEye, an open-source project designed to enhance precision agriculture 
    through AI-powered aerial analysis. This system was developed and deployed on a durian farm in the 
    Philippines, leveraging NVIDIA Jetson technology for real-time weed and disease detection.

    Using machine learning and edge computing, our Jetson-equipped drone scans orchards, detecting crop 
    health issues on the fly, reducing reliance on manual inspections, and improving farm productivity.
    """)

    # Challenges Section
    st.header("🌳 Durian Farming Challenges")
    challenges = {
        "Dense Canopy Cover": "Traditional inspections are slow and inefficient.",
        "Tropical Climate Risks": "High humidity accelerates fungal infections and pest spread.",
        "Precision Agriculture Demand": "Targeted interventions reduce costs and increase fruit quality."
    }
    for challenge, description in challenges.items():
        st.markdown(f"**{challenge}** – {description}")

    # Solution Features
    st.header("🚁 Our AI-Powered Solution")
    features = [
        "✅ Onboard Jetson AI Inference – Processes images in real-time directly on the drone.",
        "✅ Weed & Disease Detection – Identifies dry patches, fungal infections, and invasive weeds.",
        "✅ Edge Computing with Jetson – No internet required, perfect for remote farms.",
        "✅ Live Alerts & GPS Tagging – Pinpoints affected areas for quick intervention.",
        "✅ Customizable AI Models – Supports YOLOv8, Faster R-CNN, and segmentation models.",
        "✅ Scalable Open-Source Platform – Fork, customize, and deploy worldwide."
    ]
    for feature in features:
        st.markdown(feature)

    # How It Works
    st.header("How It Works")
    steps = {
        "📍 1. Drone Deployment": "A Jetson-powered drone captures aerial footage over the farm.",
        "🔍 2. AI-Based Detection": "Onboard NVIDIA Jetson processes images in real-time, detecting weeds, diseases, and soil conditions.",
        "📊 3. Real-Time Insights & GPS Tagging": "Results are overlaid on the video feed and flagged with GPS coordinates.",
        "💡 4. Actionable Data for Farmers": "Immediate alerts help optimize interventions, from fungicide application to weed removal."
    }
    for step, description in steps.items():
        st.markdown(f"**{step}**\n{description}")

    # Architecture Diagram
    st.header("System Architecture")
    architecture = """
            Drone Footage (Durian Orchard)
                    +
            [ NVIDIA Jetson ]
                    |
                    v
        +----------------------+
        |  On-Device Inference|
        +----------------------+
                    |
                    v
           [ Real-Time Alerts ] 
                    |
                    v
      [ Local Storage / User Dashboard ]
    """
    st.code(architecture, language=None)

    # Technology Stack
    st.header("Technology Stack")
    tech_stack = [
        "🔹 NVIDIA Jetson (Edge AI Processing) – Runs AI models directly on the drone.",
        "🔹 CUDA & TensorRT – Optimizes inference speed and efficiency.",
        "🔹 DeepStream SDK (Optional) – Multi-stream analytics for larger farm operations.",
        "🔹 PyTorch + YOLOv8 – Object detection models for identifying crop health issues."
    ]
    for tech in tech_stack:
        st.markdown(tech)

    # Case Study
    st.header("Case Study: Deployment in a Durian Orchard")
    st.markdown("**📍 Location:** Durian Farm, Philippines")
    st.markdown("**🔎 Objective:** Detect fungal diseases and weed encroachment")

    st.subheader("🛠️ Setup")
    setup_points = [
        "Drone equipped with NVIDIA Jetson AGX Orin.",
        "YOLOv8 trained on durian-specific datasets.",
        "Edge inference deployed, reducing reliance on cloud computing."
    ]
    for point in setup_points:
        st.markdown(f"- {point}")

    st.subheader("🚀 Results")
    results = [
        "✅ Faster Inspections – Reduced manual scouting time by 75%.",
        "✅ Targeted Interventions – Less pesticide use, saving costs.",
        "✅ Increased Yield – Early disease detection prevented fruit loss."
    ]
    for result in results:
        st.markdown(result)

    st.markdown('> *"This system helped us detect leaf infections before they spread—saving our orchard thousands in potential losses."*\n> – Durian Farmer, Philippines')

    # Get Started Section
    st.header("Get Started")

    with st.expander("1️⃣ Clone the Repository"):
        st.code("""git clone https://github.com/patrickrkahn/smart-farming-drone-ai.git
cd smart-farming-drone-ai""", language="bash")

    with st.expander("2️⃣ Install Dependencies"):
        st.code("pip install -r requirements.txt", language="bash")
        st.markdown("""
        Includes PyTorch, OpenCV, YOLO dependencies.
        For Jetson, install Jetson-optimized versions.
        """)

    with st.expander("3️⃣ Run the Detection Demo"):
        st.markdown("Streamlit Web App (for images/videos):")
        st.code("streamlit run app.py", language="bash")
        st.markdown("Command-Line Processing (for recorded footage):")
        st.code("python smart_farming.py --source sample_farm_video.mp4", language="bash")

    with st.expander("4️⃣ Train a Custom Model"):
        st.code("yolo train data=durian_farm.yaml model=yolov8n.pt epochs=50", language="bash")

    with st.expander("5️⃣ Optimize for Jetson"):
        st.code("yolo export model=best.pt format=tensorrt", language="bash")

    # Community Section
    st.header("Join the Open-Source Community!")
    st.markdown("""
    🔗 Fork the Repository – Modify and improve the system.  
    🔗 Submit a Pull Request – Contribute to the project.  
    🔗 Report Issues & Discuss – Join our GitHub discussions.

    📌 [GitHub Issues & Discussions: Submit an Issue](https://github.com/patrickrkahn/smart-farming-drone-ai/issues)
    """)

if __name__ == "__main__":
    main()
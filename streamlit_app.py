import streamlit as st
import tempfile
import os
from pathlib import Path
from smart_farming import SmartFarmingDrone

def show_landing_page():
    st.title("Smart Farming Drone Imaging")
    st.subheader("An open-source project to detect weeds, dry patches, or crop diseases from aerial drone footage using NVIDIA GPU acceleration.")

    # Overview
    st.markdown("""
    ### Overview
    Durian farming in the Philippines has unique challenges—large orchard tracts, dense canopies, 
    and a tropical climate ripe for fungal diseases and pests. Smart Farming Drone Imaging harnesses 
    the power of computer vision to pinpoint issues in real-time and assist local growers in maximizing 
    yield and fruit quality.

    By leveraging NVIDIA GPU technologies—particularly NVIDIA Jetson devices for on-drone edge processing—this 
    project aims to minimize costly delays in identifying problem areas. Our approach is open-source, enabling 
    farmers, researchers, and developers to collaborate, adapt it for local conditions, and continually 
    improve model performance.
    """)

    # Key Features
    st.markdown("""
    ### Key Features
    - **Real-Time Durian Orchard Scans**: Process aerial footage in-flight to detect early signs of leaf 
      discoloration, fungal diseases, or weed encroachment.
    - **Edge Deployment with Jetson**: Reduce bandwidth and power consumption by running inference directly 
      on an NVIDIA Jetson-equipped drone.
    - **Modular Architecture**: Swap YOLO, Faster R-CNN, or segmentation models based on the specific farming need.
    - **Scalable & Extensible**: Integrate additional sensors (e.g., thermal, multispectral) to further refine detection.
    - **Open-Source Collaboration**: Fork, modify, and share improvements for broader adoption in Southeast Asian 
      agriculture and beyond.
    """)

    # Try Demo Button
    if st.button("Try Demo Application →", type="primary"):
        st.session_state.show_demo = True
        st.experimental_rerun()

    # Technology Stack
    st.markdown("""
    ### NVIDIA Technology Stack
    - **NVIDIA Jetson (Edge Deployment)**: A powerful, compact computing platform ideal for in-field AI inference, 
      reducing latency and reliance on internet connectivity.
    - **CUDA (Compute Unified Device Architecture)**: Parallel computing on NVIDIA GPUs accelerates both training 
      (in the lab) and inference (in the field).
    - **TensorRT (Optional)**: Optimize your trained models for the Jetson environment, achieving higher 
      frames-per-second with lower latency.
    - **DeepStream (Optional for Multi-Stream Analytics)**: Particularly useful if you have multiple drones or 
      camera feeds for larger orchard operations.
    """)

    # Future Enhancements
    st.markdown("""
    ### Future Enhancements
    - **Disease Segmentation**: Move beyond bounding boxes to precisely delineate infected leaf regions for 
      more accurate fungicide targeting.
    - **Multi-Drone Coordination**: If you have several Jetson-powered drones, share flight data in real-time 
      for larger orchard coverage.
    - **Offline Data Analysis**: Sync orchard health metrics to a local server or cloud when internet is 
      available, enabling advanced analytics.
    - **Integration with Other Sensors**: Merge data from infrared, thermal, or hyperspectral cameras to 
      detect stress levels and water deficiencies.
    """)

    # Footer with links
    st.markdown("""
    ---
    ### Community & Support
    - [GitHub Discussions/Issues](https://github.com/yourusername/smart-farming-drone-imaging/issues)
    - [Share your farm trials](https://github.com/yourusername/smart-farming-drone-imaging/discussions)
    - Tag us with #SmartFarmingAI
    """)

def show_demo_page():
    st.title("Smart Farming Drone Analysis Demo")

    # Back to Home button
    if st.button("← Back to Home"):
        st.session_state.show_demo = False
        st.experimental_rerun()

    st.write("Upload drone footage for analysis of crops and potential issues")

    # Initialize session state for the detector
    if 'detector' not in st.session_state:
        st.session_state.detector = SmartFarmingDrone()
        st.session_state.processed_video = None

    # File uploader
    uploaded_file = st.file_uploader("Upload drone footage", type=['mp4', 'avi'])

    if uploaded_file is not None:
        # Create a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Process button
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button('Process Video'):
                with st.spinner('Processing video...'):
                    output_path = "streamlit_output.mp4"
                    success = st.session_state.detector.process_video(video_path, output_path)

                    if success and os.path.exists(output_path):
                        st.session_state.processed_video = output_path
                        st.success("Video processed successfully!")
                    else:
                        st.error("Error processing video")

        # Display upload info
        with col2:
            st.info(f"Selected video: {uploaded_file.name}")

        # Display processed video if available
        if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
            st.subheader("Processed Video with Detections")
            video_file = open(st.session_state.processed_video, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()

        # Cleanup temporary file
        Path(video_path).unlink(missing_ok=True)

def main():
    st.set_page_config(
        page_title="Smart Farming Drone Imaging",
        page_icon="🌾",
        layout="wide"
    )

    # Initialize session state
    if 'show_demo' not in st.session_state:
        st.session_state.show_demo = False

    # Show either landing page or demo page
    if st.session_state.show_demo:
        show_demo_page()
    else:
        show_landing_page()

if __name__ == "__main__":
    main()
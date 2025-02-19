import streamlit as st
import tempfile
import os
from pathlib import Path
from smart_farming import SmartFarmingDrone

def show_landing_page():
    st.title("Smart Farming Drone Imaging")
    st.subheader("An open-source project for agricultural drone footage analysis")

    # Overview
    st.markdown("""
    ### Overview
    Smart Farming Drone Imaging harnesses the power of computer vision to analyze aerial drone footage
    of agricultural land, helping farmers identify issues in real-time and maximize yield. Our focus is
    on providing an open-source solution that enables farmers, researchers, and developers to collaborate
    and improve agricultural monitoring.
    """)

    # Key Features
    st.markdown("""
    ### Key Features
    - **Real-Time Analysis**: Process aerial footage to detect issues like:
        - Crop diseases
        - Weed infestations
        - Dry patches
        - Unhealthy vegetation
    - **GPU Acceleration**: Leveraging NVIDIA technology for faster processing
    - **Open Source**: Collaborate, modify, and improve the system
    - **Modular Design**: Easy to extend with new detection models
    """)

    # Try Demo Button
    if st.button("Try Demo Application →", type="primary"):
        st.session_state.show_demo = True
        st.experimental_rerun()

    # Technology Stack
    st.markdown("""
    ### Technology Stack
    - **Computer Vision**: OpenCV, YOLO object detection
    - **Deep Learning**: PyTorch, Ultralytics YOLOv8
    - **Interface**: Streamlit web application
    """)

    # Future Enhancements
    st.markdown("""
    ### Future Enhancements
    - Custom model training for crop-specific detection
    - TensorRT optimization for improved performance
    - Support for multiple video formats
    - Advanced visualization options
    """)

    # Footer with links
    st.markdown("""
    ---
    ### Get Involved
    - [View Source Code](https://github.com/yourusername/smart-farming-drone-imaging)
    - [Report Issues](https://github.com/yourusername/smart-farming-drone-imaging/issues)
    - [Documentation](https://github.com/yourusername/smart-farming-drone-imaging/wiki)
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
        page_icon="🚀",
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
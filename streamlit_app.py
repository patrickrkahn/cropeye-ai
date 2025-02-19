import streamlit as st
import os
from smart_farming import SmartFarmingDrone
import tempfile
from pathlib import Path

def set_page_config():
    st.set_page_config(
        page_title="CropEye",
        page_icon="🚁",
        layout="wide"
    )

def process_video(uploaded_file):
    # Create temporary file to save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        input_path = tmp_file.name

    try:
        # Initialize detector
        detector = SmartFarmingDrone()

        # Process the video
        output_path = "streamlit_output.mp4"
        success = detector.process_video(input_path, output_path)

        if success and os.path.exists(output_path):
            return output_path
        else:
            st.error("Error processing video")
            return None
    finally:
        # Cleanup temporary file
        os.unlink(input_path)

def main():
    set_page_config()

    st.title("CropEye")
    st.subheader("AI-Powered Drone Footage Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload drone footage", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        # Show video details
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")

        # Process button
        if st.button("Process Video"):
            with st.spinner("Processing video... This may take a few minutes."):
                output_path = process_video(uploaded_file)

                if output_path and os.path.exists(output_path):
                    st.success("Video processed successfully!")

                    # Display processed video
                    st.video(output_path)

                    # Download button
                    with open(output_path, 'rb') as file:
                        btn = st.download_button(
                            label="Download processed video",
                            data=file,
                            file_name="processed_drone_footage.mp4",
                            mime="video/mp4"
                        )

    # Instructions
    st.markdown("""
    ### Instructions:
    1. Upload your drone footage video file (MP4, AVI, or MOV format)
    2. Click 'Process Video' to start the analysis
    3. Wait for the processing to complete
    4. View the results with detected objects highlighted
    5. Download the processed video if desired

    The system will analyze the footage using YOLOv8 to detect and highlight objects of interest.
    """)

if __name__ == "__main__":
    main()
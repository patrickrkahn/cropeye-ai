import streamlit as st
import tempfile
import os
from pathlib import Path
from smart_farming import SmartFarmingDrone

st.set_page_config(page_title="Smart Farming Drone Analysis", layout="wide")

def main():
    st.title("Smart Farming Drone Analysis")
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
        if st.button('Process Video'):
            with st.spinner('Processing video...'):
                output_path = "streamlit_output.mp4"
                success = st.session_state.detector.process_video(video_path, output_path)
                
                if success:
                    st.session_state.processed_video = output_path
                    st.success("Video processed successfully!")
                else:
                    st.error("Error processing video")

        # Display processed video if available
        if st.session_state.processed_video and os.path.exists(st.session_state.processed_video):
            st.subheader("Processed Video")
            st.video(st.session_state.processed_video)

        # Cleanup temporary file
        Path(video_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()

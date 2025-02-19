import streamlit as st
import os
from smart_farming import SmartFarmingDrone
import tempfile
from pathlib import Path

def set_page_config():
    st.set_page_config(
        page_title="CropEye AI",
        page_icon="🚁",
        layout="wide"
    )

def process_video(uploaded_file):
    """Process uploaded video and return the output path"""
    # Create temporary file to save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        input_path = tmp_file.name

    try:
        # Initialize detector
        detector = SmartFarmingDrone()

        # Process the video
        output_path = Path("streamlit_output.mp4")

        # Create a progress bar
        progress_text = "Processing video... Please wait."
        progress_bar = st.progress(0, text=progress_text)

        # Process video with progress updates
        success = detector.process_video(input_path, str(output_path))

        if success and output_path.exists():
            progress_bar.progress(100, text="Processing complete!")
            return str(output_path)
        else:
            st.error("Error processing video. Please check the logs for details.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    finally:
        # Cleanup temporary file
        try:
            os.unlink(input_path)
        except Exception as e:
            st.warning(f"Could not remove temporary file: {str(e)}")

def main():
    set_page_config()

    st.title("CropEye AI")
    st.subheader("Smart Farming Drone Analysis")

    # Add introduction
    st.markdown("""
    Welcome to CropEye AI! This tool helps farmers analyze drone footage to:
    - Detect weeds and unhealthy crops
    - Monitor plant growth
    - Identify potential issues early
    """)

    # File uploader with clear instructions
    st.markdown("### Upload Drone Footage")
    st.markdown("Supported formats: MP4, AVI, MOV")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov'],
        help="Upload your drone footage for analysis"
    )

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

        # Process button with clear status updates
        if st.button("Analyze Video", help="Click to start processing the video"):
            with st.spinner("Initializing analysis..."):
                output_path = process_video(uploaded_file)

                if output_path and os.path.exists(output_path):
                    st.success("Analysis completed successfully!")

                    # Display results section
                    st.markdown("### Analysis Results")

                    # Display processed video
                    st.video(output_path)

                    # Download button
                    with open(output_path, 'rb') as file:
                        st.download_button(
                            label="Download Analyzed Video",
                            data=file,
                            file_name="analyzed_drone_footage.mp4",
                            mime="video/mp4",
                            help="Download the processed video with detections"
                        )

    # Instructions and documentation
    with st.expander("How to Use"):
        st.markdown("""
        1. **Upload Video**: Click 'Browse files' to upload your drone footage
        2. **Process**: Click 'Analyze Video' to start the analysis
        3. **Review**: Watch the processed video with detected objects highlighted
        4. **Download**: Save the analyzed video for your records

        For best results:
        - Use clear, well-lit footage
        - Keep the drone at a consistent height
        - Ensure stable flight conditions
        """)

if __name__ == "__main__":
    main()
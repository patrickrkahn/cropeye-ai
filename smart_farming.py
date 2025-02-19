import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import os
import logging
import argparse
from datetime import datetime

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'smart_farming_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartFarmingDrone:
    """
    Smart Farming Drone Analysis System
    Performs real-time detection of crops, weeds, and diseases using YOLOv8
    """

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        """
        Initialize the Smart Farming Drone analysis system

        Args:
            model_path (str): Path to the YOLO model file
            confidence_threshold (float): Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        logger.info(f"Initializing SmartFarmingDrone with model: {model_path}")

        # Validate model file
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not model_path.is_file():
            raise ValueError(f"Model path is not a file: {model_path}")

        try:
            logger.info("Loading YOLO model...")
            self.model = YOLO(str(model_path))
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            logger.info(f"Using CUDA: {torch.cuda.is_available()}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def validate_video_file(self, video_path):
        """Validate video file before processing"""
        if not isinstance(video_path, (str, Path)):
            raise ValueError("Video path must be a string or Path object")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not video_path.is_file():
            raise ValueError(f"Video path is not a file: {video_path}")

        # Check if file is readable
        try:
            with open(video_path, 'rb') as f:
                pass
        except Exception as e:
            raise IOError(f"Cannot read video file: {str(e)}")

        return video_path

    def process_frame(self, frame):
        """
        Process a single frame with object detection

        Args:
            frame (numpy.ndarray): Input frame to process

        Returns:
            numpy.ndarray: Annotated frame with detections
        """
        if frame is None:
            logger.warning("Received empty frame")
            return None

        try:
            # Ensure frame is in the correct format
            if not isinstance(frame, np.ndarray):
                raise ValueError("Frame must be a numpy array")
            if len(frame.shape) != 3:
                raise ValueError("Frame must be a 3-dimensional array (height, width, channels)")

            results = self.model(frame, verbose=False)[0]
            annotated_frame = frame.copy()
            detections_in_frame = 0

            for r in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = r
                if score < self.confidence_threshold:
                    continue

                detections_in_frame += 1
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_name = results.names[int(class_id)]

                # Draw bounding box with thicker lines
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Add label with better visibility
                label = f"{class_name}: {score:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            if detections_in_frame > 0:
                logger.debug(f"Found {detections_in_frame} objects in current frame")

            return annotated_frame

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame

    def process_video(self, input_path, output_path="output_video.mp4"):
        """
        Process video file and save the results

        Args:
            input_path (str): Path to input video file
            output_path (str): Path to save the processed video

        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            # Validate input video
            input_path = self.validate_video_file(input_path)
            logger.info(f"Processing video: {input_path}")

            # Open video capture
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise IOError("Error opening video file")

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Video properties - Resolution: {frame_width}x{frame_height}, FPS: {fps}")
            logger.info(f"Total frames to process: {total_frames}")

            # Create video writer with fallback codecs
            codecs = ['mp4v', 'avc1', 'XVID']
            out = None

            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    temp_output = str(Path(output_path))
                    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

                    if out is not None and out.isOpened():
                        logger.info(f"Using codec: {codec}")
                        break
                except Exception as e:
                    logger.warning(f"Failed with codec {codec}: {str(e)}")
                    if out is not None:
                        out.release()

            if out is None or not out.isOpened():
                raise RuntimeError("Failed to create video writer with any codec")

            # Process frames with progress bar
            frames_processed = 0
            with tqdm(total=total_frames, desc="Processing") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame = self.process_frame(frame)
                    if processed_frame is not None:
                        out.write(processed_frame)
                        frames_processed += 1
                    pbar.update(1)

            logger.info(f"Video processing completed. Processed {frames_processed}/{total_frames} frames")
            logger.info(f"Output saved to: {output_path}")

            # Verify output file
            output_path = Path(output_path)
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Output file size: {output_path.stat().st_size:,} bytes")
            else:
                raise RuntimeError("Output file is empty or was not created")

            return True

        except Exception as e:
            logger.error(f"Error during video processing: {str(e)}")
            return False

        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Smart Farming Drone Analysis")
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--output', default='output_video.mp4', help='Path to output video file')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    return parser.parse_args()

def main():
    """Main function to run the Smart Farming Drone analysis"""
    args = parse_args()

    try:
        logger.info("Initializing Smart Farming Drone system...")
        detector = SmartFarmingDrone(model_path=args.model, confidence_threshold=args.confidence)

        success = detector.process_video(args.input, args.output)

        if success:
            logger.info("Processing completed successfully!")
            print(f"\nProcessed video saved to: {args.output}")
            return 0
        else:
            logger.error("Processing failed!")
            return 1

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartFarmingDrone:
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5):
        """
        Initialize the Smart Farming Drone analysis system
        """
        self.confidence_threshold = confidence_threshold
        logger.info(f"Initializing SmartFarmingDrone with model: {model_name}")
        logger.info(f"Confidence threshold set to: {confidence_threshold}")

        try:
            logger.info("Loading YOLO model...")
            self.model = YOLO(model_name)
            logger.info(f"Model {model_name} loaded successfully")
            logger.info(f"Using device: {self.model.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            sys.exit(1)

    def process_frame(self, frame):
        """Process a single frame with object detection"""
        if frame is None:
            logger.warning("Received empty frame")
            return None

        try:
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
                logger.info(f"Found {detections_in_frame} objects in current frame")

            return annotated_frame

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame

    def process_video(self, video_path, output_path="output_video.mp4"):
        """Process video file and save the results"""
        if not Path(video_path).exists():
            logger.error(f"Error: Video file '{video_path}' not found")
            return False

        logger.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Error opening video file")
            return False

        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}, Total frames: {total_frames}")

            # Try different codecs in order of preference
            codecs = ['mp4v', 'XVID', 'avc1']
            out = None

            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

                    if out.isOpened():
                        logger.info(f"Successfully created video writer with codec: {codec}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to create video writer with codec {codec}: {str(e)}")
                    if out is not None:
                        out.release()

            if out is None or not out.isOpened():
                logger.error("Failed to create video writer with any codec")
                return False

            print(f"\nProcessing video: {video_path}")
            print(f"Input video details: {frame_width}x{frame_height} @ {fps}fps")
            print("Progress will be shown below. Processing frames...")

            pbar = tqdm(total=total_frames, desc="Processing frames")
            frames_processed = 0
            frames_written = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Reached end of video file")
                    break

                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    try:
                        out.write(processed_frame)
                        frames_written += 1
                        if frames_written % 10 == 0:  # Log every 10 frames
                            logger.info(f"Successfully written {frames_written} frames")
                    except Exception as e:
                        logger.error(f"Error writing frame {frames_processed}: {str(e)}")
                    frames_processed += 1

                pbar.update(1)

            pbar.close()
            logger.info(f"\nProcessing completed! Processed {frames_processed}/{total_frames} frames")
            logger.info(f"Successfully written {frames_written} frames")
            logger.info(f"Processed video saved to: {output_path}")

            # Verify output file exists and has size
            if Path(output_path).exists():
                file_size = Path(output_path).stat().st_size
                logger.info(f"Output file size: {file_size} bytes")
                if file_size < 1000:  # Less than 1KB
                    logger.error(f"Warning: Output file seems too small ({file_size} bytes)")
            else:
                logger.error("Output file was not created!")

            return True

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return False
        finally:
            cap.release()
            if 'out' in locals() and out is not None:
                out.release()
                logger.info("Released video writer")

def main():
    """Main function to run the Smart Farming Drone demo"""
    logger.info("Starting Smart Farming Drone demo")
    detector = SmartFarmingDrone()
    video_path = "farm_footage.mp4"
    success = detector.process_video(video_path)

    if success:
        logger.info("\nVideo processing completed successfully!")
        print("The processed video has been saved as 'output_video.mp4'")
    else:
        logger.error("\nVideo processing failed!")

if __name__ == "__main__":
    main()
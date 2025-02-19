import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import os

class SmartFarmingDrone:
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5, display_output=True):
        """
        Initialize the Smart Farming Drone analysis system

        Args:
            model_name (str): Name or path of the YOLO model to use
            confidence_threshold (float): Minimum confidence for detections
            display_output (bool): Whether to display frames in a window
        """
        self.confidence_threshold = confidence_threshold
        self.display_output = display_output

        # Set display environment variable for Replit
        if self.display_output:
            os.environ['DISPLAY'] = ':1'

        try:
            # Load the YOLO model
            self.model = YOLO(model_name)
            print(f"Model {model_name} loaded successfully")
            print(f"Using device: {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)

    def process_frame(self, frame):
        """Process a single frame with object detection"""
        if frame is None:
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

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f"{class_name}: {score:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if detections_in_frame > 0:
                print(f"Found {detections_in_frame} objects in current frame")

            return annotated_frame

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame

    def process_video(self, video_path, output_path="output_video.mp4"):
        """Process video file and save the results"""
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return False

        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            print(f"\nProcessing video: {video_path}")
            print(f"Input video details: {frame_width}x{frame_height} @ {fps}fps")

            if self.display_output:
                try:
                    cv2.namedWindow("Smart Farming Detection", cv2.WINDOW_AUTOSIZE)
                    print("Display window created successfully")
                except Exception as e:
                    print(f"Warning: Could not create display window: {str(e)}")
                    self.display_output = False

            pbar = tqdm(total=total_frames, desc="Processing frames")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    out.write(processed_frame)
                    if self.display_output:
                        try:
                            cv2.imshow("Smart Farming Detection", processed_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        except Exception as e:
                            print(f"Warning: Could not display frame: {str(e)}")
                            self.display_output = False

                pbar.update(1)

            pbar.close()
            if self.display_output:
                cv2.destroyAllWindows()

            print(f"\nProcessed {total_frames} frames")
            print(f"Saved processed video to: {output_path}")
            return True

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False
        finally:
            cap.release()
            if 'out' in locals():
                out.release()

def main():
    """Main function to run the Smart Farming Drone demo"""
    detector = SmartFarmingDrone(display_output=True)
    video_path = "farm_footage.mp4"
    success = detector.process_video(video_path)

    if success:
        print("\nVideo processing completed successfully!")
    else:
        print("\nVideo processing failed!")

if __name__ == "__main__":
    main()
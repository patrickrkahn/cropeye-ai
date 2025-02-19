import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
from tqdm import tqdm
import torch

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
        try:
            # Load the YOLO model
            self.model = YOLO(model_name)
            print(f"Model {model_name} loaded successfully")
            print(f"Using device: {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)

    def process_frame(self, frame):
        """
        Process a single frame with object detection

        Args:
            frame (numpy.ndarray): Input frame to process

        Returns:
            numpy.ndarray: Processed frame with annotations
        """
        if frame is None:
            return None

        try:
            # Run YOLO inference on the frame
            results = self.model(frame, verbose=False)[0]

            # Create a copy of the frame for drawing
            annotated_frame = frame.copy()

            # Process each detection
            detections_in_frame = 0
            for r in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = r

                if score < self.confidence_threshold:
                    continue

                detections_in_frame += 1
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Get class name
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
        """
        Process video file and save the results

        Args:
            video_path (str): Path to the input video file
            output_path (str): Path where the processed video will be saved
        """
        # Verify video file exists
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found")
            return False

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return False

        try:
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Process frames with progress bar
            print(f"\nProcessing video: {video_path}")
            print(f"Input video details: {frame_width}x{frame_height} @ {fps}fps")

            if self.display_output:
                cv2.namedWindow("Smart Farming Detection", cv2.WINDOW_NORMAL)

            pbar = tqdm(total=total_frames, desc="Processing frames")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    out.write(processed_frame)
                    if self.display_output:
                        cv2.imshow("Smart Farming Detection", processed_frame)
                        # Break if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

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
    """
    Main function to run the Smart Farming Drone demo
    """
    # Initialize the detector
    detector = SmartFarmingDrone(display_output=True)

    # Process the video
    video_path = "farm_footage.mp4"
    success = detector.process_video(video_path)

    if success:
        print("\nVideo processing completed successfully!")
    else:
        print("\nVideo processing failed!")

if __name__ == "__main__":
    main()

"""
PERFORMANCE OPTIMIZATION NOTES:

1. TensorRT Optimization:
   - Convert YOLO model to TensorRT format for faster inference
   - Example implementation:
     model.to('cuda').export(format='engine')

2. Batch Processing:
   - Process multiple frames simultaneously
   - Implement frame skipping for real-time processing

3. GPU Memory Management:
   - Use torch.cuda.empty_cache() periodically
   - Implement model precision reduction (FP16/INT8)

4. Custom Model Training:
   - Collect domain-specific dataset
   - Label images with specific crop/weed classes
   - Train YOLOv8 with custom dataset:
     model = YOLO('yolov8n.pt')
     model.train(data='custom_data.yaml', epochs=100)

5. Multi-Threading:
   - Separate video reading and processing threads
   - Implement frame buffer for smooth processing
"""
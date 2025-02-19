import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path

class SmartFarmingDrone:
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5):
        """
        Initialize the Smart Farming Drone analysis system
        
        Args:
            model_name (str): Name or path of the YOLO model to use
            confidence_threshold (float): Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        try:
            # Load the YOLO model
            self.model = YOLO(model_name)
            print(f"Model {model_name} loaded successfully")
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

        # Run YOLO inference on the frame
        # In a real farming scenario, replace with custom-trained model
        # trained on crop/weed dataset
        results = self.model(frame)[0]
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Process each detection
        for detection in results.boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection
            
            if confidence < self.confidence_threshold:
                continue
                
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get class name
            class_name = results.names[int(class_id)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_frame

    def process_video(self, video_path):
        """
        Process video file and display results
        
        Args:
            video_path (str): Path to the video file
        """
        # Verify video file exists
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found")
            return

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        try:
            # Get video properties for potential saving
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                processed_frame = self.process_frame(frame)
                if processed_frame is None:
                    continue

                # Display the frame
                cv2.imshow("Smart Farming Drone Demo", processed_frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error processing video: {str(e)}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

def main():
    """
    Main function to run the Smart Farming Drone demo
    """
    # Initialize the detector
    detector = SmartFarmingDrone()
    
    # Process the video
    video_path = "farm_footage.mp4"
    detector.process_video(video_path)

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

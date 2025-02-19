import cv2
import numpy as np

def create_test_video(output_path='farm_footage.mp4', duration=10, fps=30):
    """
    Create a test video with moving objects for YOLO detection
    """
    # Video settings
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create moving object parameters
    object_size = 50
    x, y = width // 4, height // 4
    dx, dy = 2, 3  # Movement speed
    
    try:
        # Generate frames
        for frame_idx in range(duration * fps):
            # Create a frame with green background (simulating grass/field)
            frame = np.ones((height, width, 3), dtype=np.uint8) * np.array([100, 150, 100], dtype=np.uint8)
            
            # Update object position
            x = (x + dx) % (width - object_size)
            y = (y + dy) % (height - object_size)
            
            # Draw "object" (rectangle)
            cv2.rectangle(frame, (x, y), (x + object_size, y + object_size), (0, 0, 255), -1)
            
            # Add frame number as text
            cv2.putText(frame, f'Frame: {frame_idx}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
        
        print(f"Test video created successfully: {output_path}")
        return True
    
    except Exception as e:
        print(f"Error creating test video: {str(e)}")
        return False
    finally:
        out.release()

if __name__ == "__main__":
    create_test_video()

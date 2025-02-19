pip install -r requirements.txt
   ```

2. Place your drone footage video file in the project directory as `farm_footage.mp4`

## Usage

Run the script with:
```bash
python smart_farming.py
```

The script will:
1. Load the YOLOv8 model
2. Process the input video frame by frame
3. Draw bounding boxes around detected objects
4. Save the annotated video as `output_video.mp4`

Progress will be displayed in the console during processing.

## Customization

To use a custom-trained model for specific agricultural detection:
1. Train YOLOv8 on your labeled dataset of weeds/crops
2. Update the model path in the script:
   ```python
   detector = SmartFarmingDrone(model_name="path_to_your_model.pt")
git clone https://github.com/yourusername/cropeye-ai.git
cd cropeye-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
Process a video file:
```bash
python smart_farming.py --input farm_footage.mp4 --output results.mp4
```

### Streamlit Web Interface
Launch the interactive web interface:
```bash
streamlit run streamlit_app.py
```

### Model Training
Train on your custom dataset:
```bash
python train.py --data path/to/data.yaml --epochs 50
```

## Project Structure
```
├── smart_farming.py      # Core detection logic
├── streamlit_app.py      # Web interface
├── requirements.txt      # Project dependencies
├── models/              # Pre-trained models
├── utils/              # Helper functions
└── data/               # Example datasets
```

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- YOLO by Ultralytics
- OpenCV community
- Streamlit team

## Citation
If you use this project in your research, please cite:
```
@software{cropeye_ai2025,
  author = {CropEye AI Team},
  title = {CropEye AI: Smart Farming Drone Analysis},
  year = {2025},
  url = {https://github.com/yourusername/cropeye-ai}
}
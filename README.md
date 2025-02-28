# Image Quantization using CUDA-Accelerated K-Means Clustering

## Overview
This project implements **K-Means Clustering** for image quantization using **CUDA acceleration** to significantly improve performance. The application is built with **PyTorch (CUDA), NumPy, OpenCV, and Tkinter** for an interactive GUI.

## Features
- **GPU-Accelerated K-Means Clustering** using PyTorch CUDA tensors.
- **Tkinter GUI** for easy image selection and quantization.
- **Adjustable Number of Colors** to control the level of quantization.
- **Performance Timer** to display execution time.
- **Supports Various Image Formats** including JPG, PNG, and BMP.

## Technologies Used
- **Programming Language**: Python 3
- **Libraries**: PyTorch, NumPy, OpenCV, Tkinter, PIL (Pillow)
- **CUDA Support**: GPU acceleration for fast clustering

## Installation
### Prerequisites
Ensure you have the following installed:
- **Python 3.10+**
- **CUDA-compatible GPU**
- **PyTorch with CUDA**
- **Required Python libraries**

### Installation Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/The-Silly-Glitch/Color-Quantization-Using-Cuda.git
   cd Color-Quantization-Using-Cuda
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Running the GUI
To start the application, run:
```sh
python gui.py
```

### Using the Application
1. Click **Browse Image** and select an image.
2. Enter the desired number of colors (clusters).
3. Click **Quantize Image** to process the image.
4. View execution time and preview the quantized image.
5. Click **Save Quantized Image** to save the output.

## Performance Improvement
| Image Size | CPU Execution Time | CUDA Execution Time |
|------------|-------------------|-------------------|
| 512×512    | ~5s               | ~0.8s             |
| 1024×768   | ~12s              | ~1.5s             |
| 1920×1080  | ~30s              | ~3.5s             |

## Folder Structure
```
├── src
│   ├── gui.py               # GUI application
│   ├── kmeans_clustering.py # CUDA-accelerated K-Means
│   ├── image_processor.py   # Image loading and processing
├── requirements.txt         # Required dependencies
├── README.md                # Project documentation
```

## Future Improvements
- Implement **Mini-Batch K-Means** for further speedup.
- Add support for **custom distance metrics**.
- Improve **color accuracy with alternative initialization techniques**.

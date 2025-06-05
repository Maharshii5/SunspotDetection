# ☀️ Sunspot Detector

Detect and visualize sunspots in solar images using Python-based image processing and FITS file analysis.

---

## 📌 Overview

This project is designed to detect sunspots from solar images (in FITS format), visualize them, and support further solar data analysis. Sunspot detection plays a crucial role in studying solar activity and its effects on Earth, such as geomagnetic storms and satellite communication disruptions.

---

## 🗂️ Project Structure

SunspotDetector/
├── results/ # Output data including annotated visualizations
│ ├── sample1_with_sunspots_spots_.fits
│ └── sample2_no_sunspots_spots_.fits
│
├── sample_data/ # Input FITS files and visualizations
│ ├── sample1_with_sunspots.fits
│ ├── sample1_visualization.png
│ ├── sample2_no_sunspots.fits
│ ├── sample2_visualization.png
│ ├── sample3_large_sunspots.fits
│ └── sample3_visualization.png
│
├── templates/ # Templates for plotting/report generation
│ └── ...
│
├── generate_sample_fits.py # Script to create sample FITS files
├── main.py # Entry point script for sunspot detection
├── models.py # Detection/processing models/methods
├── solar_analysis.py # Core analysis logic
├── solar_analysis_submission.py# Submission-ready version of the analysis
├── test_solar_analysis.py # Unit tests for the analysis
├── internshipproj.py # Related project script (for internship context)
├── pyproject.toml # Project dependencies
├── uv.lock # Virtual environment lock file
├── LICENSE # License information
└── README.md # This file


---

## 🚀 Getting Started

### 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/SunspotDetector.git
cd SunspotDetector
Set up a virtual environment and install dependencies:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt  # If using requirements
# or
pip install .  # If using pyproject.toml
🖼️ Run the Detector
bash
Copy
Edit
python main.py --input sample_data/sample1_with_sunspots.fits
This will generate a sunspot visualization in the results/ directory.

# Solar Image Analysis Tool

This Python application detects the Sun's limb and identifies sunspots in solar telescope FITS images. It includes both a command-line interface and a web application for interactive usage.

## Features

- **Sun Limb Detection**: Identifies the visible edge of the Sun as a circle and calculates its center coordinates (x, y) and radius.
- **Sunspot Identification**: Creates a binary image marking sunspot regions (1 for sunspots, 0 elsewhere).
- **Universal Processing**: Works with multiple images without requiring user modification.
- **Adaptive Thresholding**: Uses multi-stage adaptive thresholding instead of a single intensity threshold.
- **Interactive Web Interface**: Upload and process FITS files through a user-friendly web application.
- **Visualization**: Generates visual representations of the Sun's limb and detected sunspots.

## Requirements

- Python 3.7+
- Required Python packages:
  - astropy (for FITS file handling)
  - numpy (for numerical operations)
  - scipy (for image processing and optimization)
  - matplotlib (for visualization)
  - flask (for web interface)
  - gunicorn (for web server)

## Usage

### Command-Line Interface

```bash
python solar_analysis.py path/to/fits_file.fits
```

Options:
- `-o, --output-dir`: Directory to save outputs (default: results)
- `-v, --visualize`: Generate visualizations of the results

### Web Application

To run the web application:

```bash
python main.py
```

or with gunicorn:

```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port main:app
```

Then open your browser and navigate to: `http://localhost:5000`

### Sample Data

The repository includes a script to generate synthetic FITS files for testing:

```bash
python generate_sample_fits.py
```

This creates three sample FITS files in the `sample_data` directory:
- `sample1_with_sunspots.fits`: Solar disk with multiple sunspots
- `sample2_no_sunspots.fits`: Clean solar disk without sunspots
- `sample3_large_sunspots.fits`: Solar disk with a few large sunspots

These synthetic files can be used to test both the command-line interface and web application.

## Project Structure

- `solar_analysis.py`: Core functionality for solar image processing
- `main.py`: Flask web application
- `templates/`: HTML templates for the web interface
- `test_solar_analysis.py`: Unit tests

## Algorithm

The application uses the following steps to process solar images:

1. **Preprocessing**: Normalize the image and apply contrast enhancement
2. **Limb Detection**:
   - Apply edge detection to identify the Sun's edge
   - Use circular Hough transform to find the best-fitting circle
   - Calculate the Sun's center coordinates and radius
3. **Sunspot Detection**:
   - Apply regional contrast enhancement
   - Use adaptive thresholding to identify dark regions (potential sunspots)
   - Apply morphological operations to clean up the binary mask
   - Label connected components to count sunspot regions

## Example Output

For each processed image, the application provides:
- Sun's center coordinates (x, y) in pixels
- Sun's radius in pixels
- Binary mask identifying sunspot regions
- Number of detected sunspot regions
- Visualization of the results (optional)

## Author

This application was created for the PRL Summer Internship Programme, March 2025.

## License

This project is open source and available under the MIT License.

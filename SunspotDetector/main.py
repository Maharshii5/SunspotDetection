#!/usr/bin/env python3
"""
Solar Image Analysis Web Application

A Flask web interface for uploading and processing solar telescope FITS images
to detect the Sun's limb and identify sunspots.

March 2025
"""

import os
import io
import uuid
import logging
from datetime import datetime
from flask import (
    Flask, flash, request, redirect, url_for, render_template, 
    send_from_directory, session, abort
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from astropy.io import fits

# Import solar analysis module and database models
import solar_analysis
from models import db, ProcessedImage

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.secret_key = os.environ.get("SESSION_SECRET", "solar-analysis-secret-key")

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///solar_analysis.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Initialize database tables
with app.app_context():
    db.create_all()
    logger.info("Database tables created")

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'fits', 'fit'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Homepage route."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    # Check if the file is allowed
    if not allowed_file(file.filename):
        flash('Only FITS files (.fits, .fit) are allowed', 'error')
        return redirect(url_for('index'))
    
    try:
        # Generate a unique filename to prevent collisions
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the uploaded file
        file.save(filepath)
        logger.info(f"File saved at {filepath}")
        
        # Process the solar image
        results = solar_analysis.process_solar_image(
            filepath, 
            output_dir=app.config['RESULTS_FOLDER'],
            visualize=True
        )
        
        # Get base filename without extension
        base_filename = os.path.splitext(unique_filename)[0]
        
        # Generate visualization
        visualization_path = generate_web_visualization(
            results['center_x'],
            results['center_y'],
            results['radius'],
            results['sunspots_mask'],
            results['image'],
            output_path=os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_visualization.png")
        )
        
        # Calculate number of sunspot regions
        from scipy import ndimage
        labeled_array, num_sunspots = ndimage.label(results['sunspots_mask'])
        
        # Save results to database
        processed_image = ProcessedImage(
            filename=unique_filename,
            original_filename=file.filename,
            center_x=float(results['center_x']),
            center_y=float(results['center_y']),
            radius=float(results['radius']),
            num_sunspots=int(num_sunspots),
            visualization_path=os.path.basename(visualization_path),
            mask_path=base_filename + '_sunspots_mask.fits'
        )
        
        # Add and commit to database
        with app.app_context():
            db.session.add(processed_image)
            db.session.commit()
            logger.info(f"Saved results to database, ID: {processed_image.id}")
        
        # Store results in session
        session['processing_results'] = {
            'filename': unique_filename,
            'center_x': float(results['center_x']),
            'center_y': float(results['center_y']),
            'radius': float(results['radius']),
            'sunspots_mask': base_filename + '_sunspots_mask.fits',
            'num_sunspots': int(num_sunspots),
            'visualization': os.path.basename(visualization_path)
        }
        
        # Redirect to results page
        return redirect(url_for('results'))
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        flash(f"Error processing file: {str(e)}", 'error')
        return redirect(url_for('index'))

def generate_web_visualization(center_x, center_y, radius, sunspots_mask, original_image=None, output_path=None):
    """
    Generate a visualization of the results suitable for web display.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # If original image is provided, use it as background
    if original_image is not None:
        ax.imshow(original_image, cmap='gray')
    else:
        # Create a blank canvas the same size as the mask
        ax.imshow(np.zeros_like(sunspots_mask, dtype=float), cmap='gray')
    
    # Plot the sun's limb
    circle = plt.Circle((center_x, center_y), radius, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(circle)
    
    # Create a colored overlay of the sunspots
    mask_colored = np.zeros((*sunspots_mask.shape, 4))  # RGBA
    mask_colored[sunspots_mask, 0] = 1.0  # Red
    mask_colored[sunspots_mask, 3] = 0.7  # Alpha
    
    ax.imshow(mask_colored)
    
    # Add labels and annotations
    ax.set_title('Solar Analysis Results', fontsize=16)
    ax.text(0.02, 0.98, f"Center: ({center_x:.1f}, {center_y:.1f})", 
            transform=ax.transAxes, color='white', fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(0.02, 0.93, f"Radius: {radius:.1f} pixels", 
            transform=ax.transAxes, color='white', fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Count sunspot regions
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(sunspots_mask)
    ax.text(0.02, 0.88, f"Sunspot regions: {num_features}", 
            transform=ax.transAxes, color='white', fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_axis_off()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        img_data = io.BytesIO()
        FigureCanvas(fig).print_png(img_data)
        plt.close(fig)
        return img_data.getvalue()

@app.route('/results')
def results():
    """Display processing results."""
    # Check if we have results in the session
    if 'processing_results' not in session:
        flash('No processing results found. Please upload an image first.', 'warning')
        return redirect(url_for('index'))
    
    # Get visualization file path
    results = session['processing_results']
    visualization_path = results.get('visualization')
    
    return render_template('results.html', 
                           results=results,
                           visualization_path=visualization_path)

@app.route('/history')
def history():
    """Display history of processed images."""
    # Get all processed images from database, ordered by most recent first
    processed_images = ProcessedImage.query.order_by(ProcessedImage.processed_at.desc()).all()
    
    return render_template('history.html', processed_images=processed_images)

@app.route('/download/<filename>')
def download_file(filename):
    """Download a processed file."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)

@app.route('/visualization/<filename>')
def show_visualization(filename):
    """Display a visualization image."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    flash('File too large. Maximum size is 16 MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
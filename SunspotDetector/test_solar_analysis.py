#!/usr/bin/env python3
"""
Unit tests for the solar_analysis module.

This script tests the functionality of the solar image analysis code
to ensure it correctly detects the Sun's limb and sunspots.
"""

import unittest
import numpy as np
from astropy.io import fits
import os
import tempfile

import solar_analysis


class TestSolarAnalysis(unittest.TestCase):
    """Test cases for solar analysis functions."""
    
    def setUp(self):
        """Create synthetic test images for testing."""
        # Create a simple synthetic solar image
        self.image_size = 200
        self.center_x, self.center_y = 100, 100
        self.radius = 80
        
        # Create coordinate grids
        y, x = np.ogrid[:self.image_size, :self.image_size]
        dist_from_center = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        
        # Create disk with limb darkening effect
        self.sun_image = np.zeros((self.image_size, self.image_size))
        solar_disk = dist_from_center <= self.radius
        
        # Apply limb darkening effect (brighter in center, darker at edges)
        limb_darkening = 1 - 0.5 * (dist_from_center / self.radius)**2
        limb_darkening[~solar_disk] = 0
        self.sun_image = 0.1 + 0.9 * limb_darkening
        
        # Add some noise
        self.sun_image += np.random.normal(0, 0.01, self.sun_image.shape)
        self.sun_image = np.clip(self.sun_image, 0, 1)
        
        # Create a version with sunspots
        self.sun_with_spots = self.sun_image.copy()
        
        # Add a few synthetic sunspots
        # Make a larger sunspot group
        spot_center_x, spot_center_y = self.center_x - 30, self.center_y + 20
        spot_radius = 8
        spot_mask = (x - spot_center_x)**2 + (y - spot_center_y)**2 <= spot_radius**2
        self.sun_with_spots[spot_mask] *= 0.3  # Darken the region
        
        # Add a smaller spot
        spot_center_x, spot_center_y = self.center_x + 35, self.center_y - 15
        spot_radius = 4
        spot_mask = (x - spot_center_x)**2 + (y - spot_center_y)**2 <= spot_radius**2
        self.sun_with_spots[spot_mask] *= 0.4  # Darken the region
        
        # Save temporary FITS files for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Save sun without spots
        self.fits_no_spots = os.path.join(self.temp_dir, "sun_no_spots.fits")
        hdu = fits.PrimaryHDU(self.sun_image)
        hdu.header['TELESCOP'] = 'TEST'
        hdu.writeto(self.fits_no_spots, overwrite=True)
        
        # Save sun with spots
        self.fits_with_spots = os.path.join(self.temp_dir, "sun_with_spots.fits")
        hdu = fits.PrimaryHDU(self.sun_with_spots)
        hdu.header['TELESCOP'] = 'TEST'
        hdu.writeto(self.fits_with_spots, overwrite=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.fits_no_spots):
            os.remove(self.fits_no_spots)
        if os.path.exists(self.fits_with_spots):
            os.remove(self.fits_with_spots)
        os.rmdir(self.temp_dir)
    
    def test_read_fits_image(self):
        """Test reading FITS images."""
        image_data, header = solar_analysis.read_fits_image(self.fits_no_spots)
        self.assertIsNotNone(image_data)
        self.assertIsNotNone(header)
        self.assertEqual(image_data.shape, (self.image_size, self.image_size))
        self.assertEqual(header['TELESCOP'], 'TEST')
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        image_data, _ = solar_analysis.read_fits_image(self.fits_no_spots)
        processed = solar_analysis.preprocess_image(image_data)
        self.assertEqual(processed.shape, image_data.shape)
        # Preprocessed image should be in [0, 1] range
        self.assertTrue(0 <= np.min(processed) <= np.max(processed) <= 1)
    
    def test_limb_detection_no_spots(self):
        """Test limb detection on image without sunspots."""
        image_data, _ = solar_analysis.read_fits_image(self.fits_no_spots)
        processed = solar_analysis.preprocess_image(image_data)
        
        center_x, center_y, radius, _ = solar_analysis.detect_sun_limb(processed)
        
        # Check if detected center and radius are close to the expected values
        self.assertAlmostEqual(center_x, self.center_x, delta=2)
        self.assertAlmostEqual(center_y, self.center_y, delta=2)
        self.assertAlmostEqual(radius, self.radius, delta=2)
    
    def test_limb_detection_with_spots(self):
        """Test limb detection on image with sunspots."""
        image_data, _ = solar_analysis.read_fits_image(self.fits_with_spots)
        processed = solar_analysis.preprocess_image(image_data)
        
        center_x, center_y, radius, _ = solar_analysis.detect_sun_limb(processed)
        
        # Check if detected center and radius are close to the expected values
        self.assertAlmostEqual(center_x, self.center_x, delta=2)
        self.assertAlmostEqual(center_y, self.center_y, delta=2)
        self.assertAlmostEqual(radius, self.radius, delta=2)
    
    def test_sunspot_detection_no_spots(self):
        """Test sunspot detection on image without sunspots."""
        image_data, _ = solar_analysis.read_fits_image(self.fits_no_spots)
        processed = solar_analysis.preprocess_image(image_data)
        
        center_x, center_y, radius, _ = solar_analysis.detect_sun_limb(processed)
        sunspots_mask = solar_analysis.detect_sunspots(processed, center_x, center_y, radius)
        
        # There should be very few or no detected sunspots in this image
        # Count the number of sunspot pixels
        num_sunspot_pixels = np.sum(sunspots_mask)
        
        # This threshold may need adjustment but should be very low
        self.assertLess(num_sunspot_pixels, 50)
    
    def test_sunspot_detection_with_spots(self):
        """Test sunspot detection on image with sunspots."""
        image_data, _ = solar_analysis.read_fits_image(self.fits_with_spots)
        processed = solar_analysis.preprocess_image(image_data)
        
        center_x, center_y, radius, _ = solar_analysis.detect_sun_limb(processed)
        sunspots_mask = solar_analysis.detect_sunspots(processed, center_x, center_y, radius)
        
        # There should be a significant number of detected sunspot pixels
        num_sunspot_pixels = np.sum(sunspots_mask)
        
        # We should detect a substantial number of sunspot pixels
        self.assertGreater(num_sunspot_pixels, 100)
        
        # Check that sunspots are detected in the expected locations
        # We need to find if the two major spot areas we created have been detected
        labeled_array, num_features = solar_analysis.ndimage.label(sunspots_mask)
        self.assertGreaterEqual(num_features, 1)  # Should detect at least one spot group
    
    def test_full_processing_pipeline(self):
        """Test the complete processing pipeline."""
        # Test on image with spots
        results = solar_analysis.process_solar_image(self.fits_with_spots, visualize=False)
        self.assertIsNotNone(results)
        self.assertIn('center_x', results)
        self.assertIn('center_y', results)
        self.assertIn('radius', results)
        self.assertIn('sunspots_mask', results)
        self.assertIn('num_sunspots', results)
        
        # Test on image without spots
        results = solar_analysis.process_solar_image(self.fits_no_spots, visualize=False)
        self.assertIsNotNone(results)
        # The number of detected sunspots should be very low
        self.assertLessEqual(results['num_sunspots'], 2)


if __name__ == '__main__':
    unittest.main()

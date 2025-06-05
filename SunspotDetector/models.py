from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ProcessedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    center_x = db.Column(db.Float, nullable=False)
    center_y = db.Column(db.Float, nullable=False)
    radius = db.Column(db.Float, nullable=False)
    num_sunspots = db.Column(db.Integer, nullable=False)
    visualization_path = db.Column(db.String(255), nullable=True)
    mask_path = db.Column(db.String(255), nullable=True)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProcessedImage {self.original_filename}>"
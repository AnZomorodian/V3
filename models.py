from app import db

class AnalysisRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    grand_prix = db.Column(db.String(100), nullable=False)
    session = db.Column(db.String(50), nullable=False)
    analysis_type = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    
    def to_dict(self):
        return {
            'id': self.id,
            'year': self.year,
            'grand_prix': self.grand_prix,
            'session': self.session,
            'analysis_type': self.analysis_type,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

import logging
from flask import render_template, jsonify, request
from app import app, db
from models import AnalysisRequest
from analytics.quantum_analytics import QuantumF1Analytics
from analytics.race_strategy import RaceStrategyAnalyzer
from analytics.real_time_analytics import RealTimeAnalyzer, LiveDataStreamer
from analytics.stress_index import DriverStressAnalyzer
from analytics.advanced_analytics import AdvancedF1Analytics

# Initialize analytics modules
quantum_analytics = QuantumF1Analytics()
strategy_analyzer = RaceStrategyAnalyzer()
realtime_analyzer = RealTimeAnalyzer()
stress_analyzer = DriverStressAnalyzer()
advanced_analytics = AdvancedF1Analytics()
live_streamer = LiveDataStreamer()

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/quantum-analysis', methods=['POST'])
def quantum_analysis():
    """Quantum performance analysis endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        session = data.get('session', 'Race')
        
        if not year or not grand_prix:
            return jsonify({'error': 'Year and grand_prix are required'}), 400
        
        # Log the request
        analysis_request = AnalysisRequest(
            year=year,
            grand_prix=grand_prix,
            session=session,
            analysis_type='quantum_analysis'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = quantum_analytics.quantum_performance_analysis(year, grand_prix, session)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in quantum analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/race-strategy', methods=['POST'])
def race_strategy():
    """Race strategy analysis endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        
        if not year or not grand_prix:
            return jsonify({'error': 'Year and grand_prix are required'}), 400
        
        # Log the request
        analysis_request = AnalysisRequest(
            year=year,
            grand_prix=grand_prix,
            session='Race',
            analysis_type='race_strategy'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = strategy_analyzer.analyze_race_strategy(year, grand_prix)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in race strategy analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime-status', methods=['POST'])
def realtime_status():
    """Real-time session status endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        
        if not year or not grand_prix:
            return jsonify({'error': 'Year and grand_prix are required'}), 400
        
        result = realtime_analyzer.get_live_session_status(year, grand_prix)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in realtime status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance-trends', methods=['POST'])
def performance_trends():
    """Performance trends analysis endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        session = data.get('session', 'Race')
        
        if not year or not grand_prix:
            return jsonify({'error': 'Year and grand_prix are required'}), 400
        
        result = realtime_analyzer.get_performance_trends(year, grand_prix, session)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in performance trends: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stress-analysis', methods=['POST'])
def stress_analysis():
    """Driver stress analysis endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        session = data.get('session', 'Race')
        driver = data.get('driver')
        
        if not year or not grand_prix or not driver:
            return jsonify({'error': 'Year, grand_prix, and driver are required'}), 400
        
        # Log the request
        analysis_request = AnalysisRequest(
            year=year,
            grand_prix=grand_prix,
            session=session,
            analysis_type=f'stress_analysis_{driver}'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = stress_analyzer.analyze_driver_stress(year, grand_prix, session, driver)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in stress analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/advanced-analysis', methods=['POST'])
def advanced_analysis():
    """Advanced session analysis endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        session = data.get('session', 'Race')
        
        if not year or not grand_prix:
            return jsonify({'error': 'Year and grand_prix are required'}), 400
        
        # Log the request
        analysis_request = AnalysisRequest(
            year=year,
            grand_prix=grand_prix,
            session=session,
            analysis_type='advanced_analysis'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = advanced_analytics.comprehensive_session_analysis(year, grand_prix, session)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in advanced analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/streaming-data', methods=['POST'])
def streaming_data():
    """Live streaming data endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        
        if not year or not grand_prix:
            return jsonify({'error': 'Year and grand_prix are required'}), 400
        
        result = live_streamer.get_streaming_data(year, grand_prix)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in streaming data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis-history', methods=['GET'])
def analysis_history():
    """Get analysis request history"""
    try:
        requests = AnalysisRequest.query.order_by(AnalysisRequest.created_at.desc()).limit(50).all()
        return jsonify([req.to_dict() for req in requests])
        
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

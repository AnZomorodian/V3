import logging
from flask import render_template, jsonify, request
from app import app, db
from models import AnalysisRequest
from analytics.quantum_analytics import QuantumF1Analytics
from analytics.race_strategy import RaceStrategyAnalyzer
from analytics.real_time_analytics import RealTimeAnalyzer, LiveDataStreamer
from analytics.stress_index import DriverStressAnalyzer
from analytics.advanced_analytics import AdvancedF1Analytics
from analytics.visualizations import (
    create_telemetry_plot, create_tire_strategy_plot, create_race_progression_plot,
    create_speed_comparison_plot, create_lap_time_distribution
)
from analytics.track_analysis import TrackAnalyzer
from analytics.tire_performance import TirePerformanceAnalyzer
from analytics.downforce_analysis import DownforceAnalyzer
from analytics.driver_comparison import DriverComparisonAnalyzer

# Initialize analytics modules
quantum_analytics = QuantumF1Analytics()
strategy_analyzer = RaceStrategyAnalyzer()
realtime_analyzer = RealTimeAnalyzer()
stress_analyzer = DriverStressAnalyzer()
advanced_analytics = AdvancedF1Analytics()
live_streamer = LiveDataStreamer()
track_analyzer = TrackAnalyzer()
tire_analyzer = TirePerformanceAnalyzer()
downforce_analyzer = DownforceAnalyzer()
driver_comparison_analyzer = DriverComparisonAnalyzer()

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

@app.route('/api/weather-analysis', methods=['POST'])
def weather_analysis():
    """Weather impact analysis endpoint"""
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
            analysis_type='weather_analysis'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = advanced_analytics.analyze_weather_impact(year, grand_prix, session)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in weather analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/driver-comparison', methods=['POST'])
def driver_comparison():
    """Driver head-to-head comparison endpoint"""
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
            analysis_type='driver_comparison'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = advanced_analytics.compare_drivers(year, grand_prix, session)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in driver comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

# New enhanced analytics endpoints
@app.route('/api/track-analysis', methods=['POST'])
def track_analysis():
    """Track characteristics and sector analysis endpoint"""
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
            analysis_type='track_analysis'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = track_analyzer.get_track_characteristics(year, grand_prix, session)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in track analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/track-mastery', methods=['POST'])
def track_mastery():
    """Driver track mastery analysis endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        session = data.get('session', 'Race')
        
        if not year or not grand_prix:
            return jsonify({'error': 'Year and grand_prix are required'}), 400
        
        result = track_analyzer.get_driver_track_mastery(year, grand_prix, session)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in track mastery analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tire-performance', methods=['POST'])
def tire_performance():
    """Tire performance analysis endpoint"""
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
            analysis_type='tire_performance'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = tire_analyzer.analyze_race_tire_performance(year, grand_prix)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in tire performance analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/downforce-analysis', methods=['POST'])
def downforce_analysis():
    """Downforce and aerodynamic analysis endpoint"""
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
            analysis_type='downforce_analysis'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = downforce_analyzer.analyze_downforce_settings(year, grand_prix, session)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in downforce analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/comprehensive-driver-comparison', methods=['POST'])
def comprehensive_driver_comparison():
    """Comprehensive driver comparison endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        session = data.get('session', 'Race')
        drivers = data.get('drivers', [])
        
        if not year or not grand_prix or not drivers:
            return jsonify({'error': 'Year, grand_prix, and drivers list are required'}), 400
        
        # Log the request
        analysis_request = AnalysisRequest(
            year=year,
            grand_prix=grand_prix,
            session=session,
            analysis_type=f'comprehensive_driver_comparison_{len(drivers)}_drivers'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = driver_comparison_analyzer.create_comprehensive_comparison(year, grand_prix, session, drivers)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in comprehensive driver comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/head-to-head-comparison', methods=['POST'])
def head_to_head_comparison():
    """Head-to-head driver comparison endpoint"""
    try:
        data = request.get_json()
        year = data.get('year')
        grand_prix = data.get('grand_prix')
        session = data.get('session', 'Race')
        driver1 = data.get('driver1')
        driver2 = data.get('driver2')
        
        if not year or not grand_prix or not driver1 or not driver2:
            return jsonify({'error': 'Year, grand_prix, driver1, and driver2 are required'}), 400
        
        # Log the request
        analysis_request = AnalysisRequest(
            year=year,
            grand_prix=grand_prix,
            session=session,
            analysis_type=f'head_to_head_{driver1}_vs_{driver2}'
        )
        db.session.add(analysis_request)
        db.session.commit()
        
        result = driver_comparison_analyzer.get_head_to_head_comparison(year, grand_prix, session, driver1, driver2)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in head-to-head comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/telemetry-visualization', methods=['POST'])
def telemetry_visualization():
    """Telemetry visualization endpoint"""
    try:
        data = request.get_json()
        telemetry_data = data.get('telemetry_data', {})
        drivers = data.get('drivers', [])
        
        if not telemetry_data or not drivers:
            return jsonify({'error': 'Telemetry data and drivers list are required'}), 400
        
        result = create_telemetry_plot(telemetry_data, drivers)
        return jsonify({'plot': result})
        
    except Exception as e:
        logger.error(f"Error in telemetry visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tire-strategy-visualization', methods=['POST'])
def tire_strategy_visualization():
    """Tire strategy visualization endpoint"""
    try:
        data = request.get_json()
        tire_data = data.get('tire_data', {})
        
        if not tire_data:
            return jsonify({'error': 'Tire data is required'}), 400
        
        result = create_tire_strategy_plot(tire_data)
        return jsonify({'plot': result})
        
    except Exception as e:
        logger.error(f"Error in tire strategy visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/race-progression-visualization', methods=['POST'])
def race_progression_visualization():
    """Race progression visualization endpoint"""
    try:
        data = request.get_json()
        lap_times_data = data.get('lap_times_data', {})
        
        if not lap_times_data:
            return jsonify({'error': 'Lap times data is required'}), 400
        
        result = create_race_progression_plot(lap_times_data)
        return jsonify({'plot': result})
        
    except Exception as e:
        logger.error(f"Error in race progression visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

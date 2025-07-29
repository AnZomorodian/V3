import pandas as pd
import numpy as np
from utils.data_loader import DataLoader

class DownforceAnalyzer:
    """Analyze downforce settings and aerodynamic performance"""
    
    def __init__(self):
        self.data_loader = DataLoader()
    
    def analyze_downforce_settings(self, year, grand_prix, session):
        """Comprehensive downforce analysis"""
        try:
            session_data = self.data_loader.load_session_data(year, grand_prix, session)
            if session_data is None:
                return None
            
            analysis = {
                'aerodynamic_efficiency': self.analyze_aerodynamic_efficiency(session_data),
                'downforce_balance': self.analyze_downforce_balance(session_data),
                'drag_analysis': self.analyze_drag_characteristics(session_data),
                'corner_speed_analysis': self.analyze_corner_speeds(session_data),
                'straight_line_performance': self.analyze_straight_line_speed(session_data),
                'setup_comparison': self.compare_aerodynamic_setups(session_data),
                'track_specific_analysis': self.analyze_track_specific_requirements(session_data, grand_prix)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_aerodynamic_efficiency(self, session_data):
        """Analyze overall aerodynamic efficiency"""
        try:
            efficiency_data = {}
            
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                fastest_lap = driver_laps.pick_fastest()
                telemetry = self.data_loader.get_telemetry_data(fastest_lap)
                
                if telemetry is None or telemetry.empty:
                    continue
                
                # Calculate aerodynamic efficiency metrics
                speed_data = telemetry['Speed']
                throttle_data = telemetry['Throttle']
                
                if speed_data.empty or throttle_data.empty:
                    continue
                
                # Analyze speed vs throttle relationship
                efficiency_metrics = self.calculate_aero_efficiency_metrics(speed_data, throttle_data)
                
                # Analyze corner vs straight performance
                corner_straight_analysis = self.analyze_corner_vs_straight_performance(telemetry)
                
                efficiency_data[driver] = {
                    'overall_efficiency_score': efficiency_metrics['efficiency_score'],
                    'speed_throttle_correlation': efficiency_metrics['correlation'],
                    'max_speed_achieved': float(speed_data.max()),
                    'avg_speed_high_throttle': efficiency_metrics['avg_speed_high_throttle'],
                    'corner_performance': corner_straight_analysis['corner_performance'],
                    'straight_performance': corner_straight_analysis['straight_performance'],
                    'efficiency_rating': self.rate_aerodynamic_efficiency(efficiency_metrics['efficiency_score'])
                }
            
            return efficiency_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_downforce_balance(self, session_data):
        """Analyze front/rear downforce balance"""
        try:
            balance_data = {}
            
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                balance_analysis = []
                
                for _, lap in driver_laps.iterrows():
                    try:
                        telemetry = self.data_loader.get_telemetry_data(lap)
                        if telemetry is None or telemetry.empty:
                            continue
                        
                        # Analyze balance through speed and throttle patterns
                        balance_metrics = self.calculate_balance_metrics(telemetry)
                        
                        if balance_metrics:
                            balance_analysis.append({
                                'lap_number': int(lap['LapNumber']),
                                'balance_score': balance_metrics['balance_score'],
                                'understeer_tendency': balance_metrics['understeer_tendency'],
                                'oversteer_tendency': balance_metrics['oversteer_tendency'],
                                'corner_entry_balance': balance_metrics['corner_entry'],
                                'corner_exit_balance': balance_metrics['corner_exit']
                            })
                    
                    except Exception:
                        continue
                
                if balance_analysis:
                    avg_balance = np.mean([ba['balance_score'] for ba in balance_analysis])
                    avg_understeer = np.mean([ba['understeer_tendency'] for ba in balance_analysis])
                    avg_oversteer = np.mean([ba['oversteer_tendency'] for ba in balance_analysis])
                    
                    balance_data[driver] = {
                        'lap_by_lap_analysis': balance_analysis,
                        'average_balance_score': float(avg_balance),
                        'understeer_tendency': float(avg_understeer),
                        'oversteer_tendency': float(avg_oversteer),
                        'balance_rating': self.rate_downforce_balance(avg_balance),
                        'setup_recommendation': self.recommend_balance_adjustment(avg_understeer, avg_oversteer)
                    }
            
            return balance_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_drag_characteristics(self, session_data):
        """Analyze drag characteristics and impact on performance"""
        try:
            drag_data = {}
            
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                drag_analysis = []
                
                for _, lap in driver_laps.iterrows():
                    try:
                        telemetry = self.data_loader.get_telemetry_data(lap)
                        if telemetry is None or telemetry.empty:
                            continue
                        
                        # Analyze drag through deceleration patterns
                        drag_metrics = self.calculate_drag_metrics(telemetry)
                        
                        if drag_metrics:
                            drag_analysis.append({
                                'lap_number': int(lap['LapNumber']),
                                'drag_coefficient_estimate': drag_metrics['drag_estimate'],
                                'deceleration_efficiency': drag_metrics['decel_efficiency'],
                                'top_speed_loss': drag_metrics['top_speed_loss'],
                                'acceleration_impact': drag_metrics['accel_impact']
                            })
                    
                    except Exception:
                        continue
                
                if drag_analysis:
                    avg_drag = np.mean([da['drag_coefficient_estimate'] for da in drag_analysis])
                    avg_top_speed_loss = np.mean([da['top_speed_loss'] for da in drag_analysis])
                    
                    drag_data[driver] = {
                        'lap_by_lap_analysis': drag_analysis,
                        'average_drag_estimate': float(avg_drag),
                        'average_top_speed_loss': float(avg_top_speed_loss),
                        'drag_efficiency_rating': self.rate_drag_efficiency(avg_drag),
                        'straight_line_impact': self.assess_straight_line_impact(avg_top_speed_loss)
                    }
            
            return drag_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_corner_speeds(self, session_data):
        """Analyze cornering speeds and downforce effectiveness"""
        try:
            corner_data = {}
            
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                fastest_lap = driver_laps.pick_fastest()
                telemetry = self.data_loader.get_telemetry_data(fastest_lap)
                
                if telemetry is None or telemetry.empty:
                    continue
                
                # Identify corner sections
                corner_sections = self.identify_corner_sections(telemetry)
                
                corner_analysis = []
                for i, section in enumerate(corner_sections):
                    corner_metrics = self.analyze_corner_section(section, i + 1)
                    if corner_metrics:
                        corner_analysis.append(corner_metrics)
                
                if corner_analysis:
                    avg_corner_speed = np.mean([ca['avg_speed'] for ca in corner_analysis])
                    min_corner_speeds = [ca['min_speed'] for ca in corner_analysis]
                    
                    corner_data[driver] = {
                        'corner_by_corner_analysis': corner_analysis,
                        'average_corner_speed': float(avg_corner_speed),
                        'slowest_corner_speed': float(min(min_corner_speeds)),
                        'fastest_corner_speed': float(max([ca['max_speed'] for ca in corner_analysis])),
                        'corner_speed_consistency': float(np.std([ca['avg_speed'] for ca in corner_analysis])),
                        'downforce_effectiveness': self.rate_downforce_effectiveness(avg_corner_speed)
                    }
            
            return corner_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_straight_line_speed(self, session_data):
        """Analyze straight-line speed performance"""
        try:
            straight_line_data = {}
            
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                fastest_lap = driver_laps.pick_fastest()
                telemetry = self.data_loader.get_telemetry_data(fastest_lap)
                
                if telemetry is None or telemetry.empty:
                    continue
                
                # Identify straight sections
                straight_sections = self.identify_straight_sections(telemetry)
                
                straight_analysis = []
                for i, section in enumerate(straight_sections):
                    straight_metrics = self.analyze_straight_section(section, i + 1)
                    if straight_metrics:
                        straight_analysis.append(straight_metrics)
                
                if straight_analysis:
                    max_speeds = [sa['max_speed'] for sa in straight_analysis]
                    acceleration_rates = [sa['acceleration_rate'] for sa in straight_analysis]
                    
                    straight_line_data[driver] = {
                        'straight_by_straight_analysis': straight_analysis,
                        'absolute_top_speed': float(max(max_speeds)),
                        'average_top_speed': float(np.mean(max_speeds)),
                        'average_acceleration_rate': float(np.mean(acceleration_rates)),
                        'straight_line_consistency': float(np.std(max_speeds)),
                        'drag_penalty_assessment': self.assess_drag_penalty(max_speeds)
                    }
            
            return straight_line_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def compare_aerodynamic_setups(self, session_data):
        """Compare aerodynamic setups between drivers"""
        try:
            setup_comparison = {}
            all_driver_data = {}
            
            # Collect data for all drivers
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                fastest_lap = driver_laps.pick_fastest()
                telemetry = self.data_loader.get_telemetry_data(fastest_lap)
                
                if telemetry is None or telemetry.empty:
                    continue
                
                all_driver_data[driver] = {
                    'max_speed': float(telemetry['Speed'].max()),
                    'avg_speed': float(telemetry['Speed'].mean()),
                    'corner_speed_avg': self.calculate_average_corner_speed(telemetry),
                    'straight_speed_avg': self.calculate_average_straight_speed(telemetry)
                }
            
            # Compare setups
            if len(all_driver_data) >= 2:
                setup_comparison['high_downforce_drivers'] = self.identify_high_downforce_drivers(all_driver_data)
                setup_comparison['low_downforce_drivers'] = self.identify_low_downforce_drivers(all_driver_data)
                setup_comparison['balanced_setups'] = self.identify_balanced_setups(all_driver_data)
                setup_comparison['setup_effectiveness'] = self.rate_setup_effectiveness(all_driver_data)
            
            return setup_comparison
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_track_specific_requirements(self, session_data, grand_prix):
        """Analyze track-specific downforce requirements"""
        try:
            # Get track characteristics
            track_requirements = self.get_track_downforce_requirements(grand_prix)
            
            # Analyze actual setups vs requirements
            actual_setups = {}
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if not driver_laps.empty:
                    fastest_lap = driver_laps.pick_fastest()
                    telemetry = self.data_loader.get_telemetry_data(fastest_lap)
                    
                    if telemetry is not None and not telemetry.empty:
                        setup_characteristics = self.estimate_setup_characteristics(telemetry)
                        actual_setups[driver] = setup_characteristics
            
            return {
                'track_requirements': track_requirements,
                'actual_setups': actual_setups,
                'setup_optimization_suggestions': self.suggest_setup_optimizations(track_requirements, actual_setups)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # Helper methods
    def calculate_aero_efficiency_metrics(self, speed_data, throttle_data):
        """Calculate aerodynamic efficiency metrics"""
        try:
            # Calculate correlation between speed and throttle
            correlation = np.corrcoef(speed_data, throttle_data)[0, 1] if len(speed_data) > 1 else 0
            
            # Calculate average speed at high throttle
            high_throttle_mask = throttle_data > 80
            avg_speed_high_throttle = speed_data[high_throttle_mask].mean() if high_throttle_mask.any() else 0
            
            # Calculate efficiency score (higher is better)
            efficiency_score = correlation * 50 + (avg_speed_high_throttle / 350) * 50
            
            return {
                'efficiency_score': float(max(0, min(100, efficiency_score))),
                'correlation': float(correlation),
                'avg_speed_high_throttle': float(avg_speed_high_throttle)
            }
        except:
            return {'efficiency_score': 50.0, 'correlation': 0.0, 'avg_speed_high_throttle': 0.0}
    
    def analyze_corner_vs_straight_performance(self, telemetry):
        """Analyze corner vs straight line performance"""
        try:
            # Identify corners (low speed zones)
            corner_mask = telemetry['Speed'] < telemetry['Speed'].quantile(0.4)
            straight_mask = telemetry['Speed'] > telemetry['Speed'].quantile(0.8)
            
            corner_performance = telemetry[corner_mask]['Speed'].mean() if corner_mask.any() else 0
            straight_performance = telemetry[straight_mask]['Speed'].mean() if straight_mask.any() else 0
            
            return {
                'corner_performance': float(corner_performance),
                'straight_performance': float(straight_performance)
            }
        except:
            return {'corner_performance': 0.0, 'straight_performance': 0.0}
    
    def calculate_balance_metrics(self, telemetry):
        """Calculate balance metrics from telemetry"""
        try:
            # Simplified balance analysis based on speed and throttle patterns
            speed_variance = telemetry['Speed'].var()
            throttle_variance = telemetry['Throttle'].var()
            
            # Calculate balance score (lower variance = better balance)
            balance_score = max(0, 100 - (speed_variance / 1000 + throttle_variance / 100))
            
            # Estimate understeer/oversteer tendencies
            understeer_tendency = max(0, min(100, throttle_variance / 10))
            oversteer_tendency = max(0, min(100, speed_variance / 1000))
            
            return {
                'balance_score': float(balance_score),
                'understeer_tendency': float(understeer_tendency),
                'oversteer_tendency': float(oversteer_tendency),
                'corner_entry': float(balance_score * 0.8),
                'corner_exit': float(balance_score * 1.2)
            }
        except:
            return None
    
    def calculate_drag_metrics(self, telemetry):
        """Calculate drag-related metrics"""
        try:
            max_speed = telemetry['Speed'].max()
            avg_speed = telemetry['Speed'].mean()
            
            # Estimate drag coefficient (simplified)
            drag_estimate = max(0, min(1, (350 - max_speed) / 100))
            
            # Calculate deceleration efficiency
            brake_zones = telemetry[telemetry['Brake'] > 0]
            if not brake_zones.empty:
                decel_efficiency = brake_zones['Speed'].std() / brake_zones['Speed'].mean()
            else:
                decel_efficiency = 0.5
            
            return {
                'drag_estimate': float(drag_estimate),
                'decel_efficiency': float(decel_efficiency),
                'top_speed_loss': float(max(0, 350 - max_speed)),
                'accel_impact': float(drag_estimate * 10)
            }
        except:
            return None
    
    def identify_corner_sections(self, telemetry):
        """Identify corner sections in telemetry"""
        try:
            # Simple corner identification based on low speeds
            corner_threshold = telemetry['Speed'].quantile(0.4)
            corner_mask = telemetry['Speed'] < corner_threshold
            
            # Group consecutive corner points
            corner_sections = []
            current_section = []
            
            for i, is_corner in enumerate(corner_mask):
                if is_corner:
                    current_section.append(i)
                else:
                    if current_section and len(current_section) > 10:  # Minimum corner length
                        corner_sections.append(telemetry.iloc[current_section])
                    current_section = []
            
            return corner_sections
        except:
            return []
    
    def analyze_corner_section(self, section, corner_number):
        """Analyze a specific corner section"""
        try:
            return {
                'corner_number': corner_number,
                'avg_speed': float(section['Speed'].mean()),
                'min_speed': float(section['Speed'].min()),
                'max_speed': float(section['Speed'].max()),
                'speed_variance': float(section['Speed'].var())
            }
        except:
            return None
    
    def identify_straight_sections(self, telemetry):
        """Identify straight sections in telemetry"""
        try:
            # Simple straight identification based on high speeds
            straight_threshold = telemetry['Speed'].quantile(0.8)
            straight_mask = telemetry['Speed'] > straight_threshold
            
            # Group consecutive straight points
            straight_sections = []
            current_section = []
            
            for i, is_straight in enumerate(straight_mask):
                if is_straight:
                    current_section.append(i)
                else:
                    if current_section and len(current_section) > 10:  # Minimum straight length
                        straight_sections.append(telemetry.iloc[current_section])
                    current_section = []
            
            return straight_sections
        except:
            return []
    
    def analyze_straight_section(self, section, straight_number):
        """Analyze a specific straight section"""
        try:
            # Calculate acceleration rate
            speeds = section['Speed'].values
            if len(speeds) > 1:
                acceleration_rate = (speeds[-1] - speeds[0]) / len(speeds)
            else:
                acceleration_rate = 0
            
            return {
                'straight_number': straight_number,
                'max_speed': float(section['Speed'].max()),
                'avg_speed': float(section['Speed'].mean()),
                'acceleration_rate': float(acceleration_rate)
            }
        except:
            return None
    
    # Rating and assessment methods
    def rate_aerodynamic_efficiency(self, efficiency_score):
        """Rate aerodynamic efficiency"""
        if efficiency_score > 80:
            return "Excellent"
        elif efficiency_score > 60:
            return "Good"
        elif efficiency_score > 40:
            return "Average"
        else:
            return "Poor"
    
    def rate_downforce_balance(self, balance_score):
        """Rate downforce balance"""
        if balance_score > 85:
            return "Well balanced"
        elif balance_score > 70:
            return "Slightly unbalanced"
        elif balance_score > 50:
            return "Moderately unbalanced"
        else:
            return "Poorly balanced"
    
    def recommend_balance_adjustment(self, understeer, oversteer):
        """Recommend balance adjustments"""
        if understeer > oversteer + 10:
            return "Reduce front downforce or increase rear downforce"
        elif oversteer > understeer + 10:
            return "Increase front downforce or reduce rear downforce"
        else:
            return "Balance appears optimal"
    
    def rate_drag_efficiency(self, drag_estimate):
        """Rate drag efficiency"""
        if drag_estimate < 0.3:
            return "Low drag - excellent straight line speed"
        elif drag_estimate < 0.6:
            return "Medium drag - balanced setup"
        else:
            return "High drag - prioritizes cornering"
    
    def assess_straight_line_impact(self, top_speed_loss):
        """Assess straight line impact"""
        if top_speed_loss < 10:
            return "Minimal impact on straight line speed"
        elif top_speed_loss < 25:
            return "Moderate impact on straight line speed"
        else:
            return "Significant impact on straight line speed"
    
    def rate_downforce_effectiveness(self, avg_corner_speed):
        """Rate downforce effectiveness in corners"""
        if avg_corner_speed > 180:
            return "High downforce - excellent cornering"
        elif avg_corner_speed > 150:
            return "Medium downforce - good cornering"
        else:
            return "Low downforce - prioritizes straight line speed"
    
    def assess_drag_penalty(self, max_speeds):
        """Assess drag penalty from setup"""
        avg_max_speed = np.mean(max_speeds)
        if avg_max_speed > 320:
            return "Low drag penalty"
        elif avg_max_speed > 300:
            return "Medium drag penalty"
        else:
            return "High drag penalty"
    
    # Additional helper methods
    def calculate_average_corner_speed(self, telemetry):
        """Calculate average corner speed"""
        try:
            corner_threshold = telemetry['Speed'].quantile(0.4)
            corner_speeds = telemetry[telemetry['Speed'] < corner_threshold]['Speed']
            return float(corner_speeds.mean()) if not corner_speeds.empty else 0
        except:
            return 0
    
    def calculate_average_straight_speed(self, telemetry):
        """Calculate average straight line speed"""
        try:
            straight_threshold = telemetry['Speed'].quantile(0.8)
            straight_speeds = telemetry[telemetry['Speed'] > straight_threshold]['Speed']
            return float(straight_speeds.mean()) if not straight_speeds.empty else 0
        except:
            return 0
    
    def identify_high_downforce_drivers(self, driver_data):
        """Identify drivers with high downforce setups"""
        drivers = []
        for driver, data in driver_data.items():
            if data['corner_speed_avg'] > data['straight_speed_avg'] * 0.6:
                drivers.append(driver)
        return drivers
    
    def identify_low_downforce_drivers(self, driver_data):
        """Identify drivers with low downforce setups"""
        drivers = []
        for driver, data in driver_data.items():
            if data['straight_speed_avg'] > data['corner_speed_avg'] * 1.8:
                drivers.append(driver)
        return drivers
    
    def identify_balanced_setups(self, driver_data):
        """Identify drivers with balanced setups"""
        drivers = []
        for driver, data in driver_data.items():
            ratio = data['corner_speed_avg'] / data['straight_speed_avg'] if data['straight_speed_avg'] > 0 else 0
            if 0.5 < ratio < 0.7:
                drivers.append(driver)
        return drivers
    
    def rate_setup_effectiveness(self, driver_data):
        """Rate overall setup effectiveness"""
        effectiveness = {}
        for driver, data in driver_data.items():
            # Simple effectiveness rating based on balanced performance
            corner_ratio = data['corner_speed_avg'] / 200 if data['corner_speed_avg'] > 0 else 0
            straight_ratio = data['straight_speed_avg'] / 300 if data['straight_speed_avg'] > 0 else 0
            effectiveness[driver] = min(100, (corner_ratio + straight_ratio) * 50)
        return effectiveness
    
    def get_track_downforce_requirements(self, grand_prix):
        """Get track-specific downforce requirements"""
        # Simplified track requirements
        high_downforce_tracks = ['Monaco', 'Hungary', 'Singapore']
        low_downforce_tracks = ['Monza', 'Spa', 'Silverstone']
        
        if grand_prix in high_downforce_tracks:
            return {
                'requirement': 'High downforce',
                'priority': 'Cornering performance',
                'setup_guidance': 'Prioritize corner speed over straight line speed'
            }
        elif grand_prix in low_downforce_tracks:
            return {
                'requirement': 'Low downforce',
                'priority': 'Straight line speed',
                'setup_guidance': 'Prioritize straight line speed and slipstream effectiveness'
            }
        else:
            return {
                'requirement': 'Medium downforce',
                'priority': 'Balanced performance',
                'setup_guidance': 'Balance between cornering and straight line performance'
            }
    
    def estimate_setup_characteristics(self, telemetry):
        """Estimate setup characteristics from telemetry"""
        try:
            corner_speed = self.calculate_average_corner_speed(telemetry)
            straight_speed = self.calculate_average_straight_speed(telemetry)
            
            if corner_speed / straight_speed > 0.6:
                return "High downforce setup"
            elif corner_speed / straight_speed < 0.4:
                return "Low downforce setup"
            else:
                return "Balanced setup"
        except:
            return "Unknown setup"
    
    def suggest_setup_optimizations(self, track_requirements, actual_setups):
        """Suggest setup optimizations"""
        suggestions = {}
        
        for driver, actual_setup in actual_setups.items():
            required = track_requirements['requirement']
            
            if required == 'High downforce' and 'Low downforce' in actual_setup:
                suggestions[driver] = "Increase downforce for better cornering performance"
            elif required == 'Low downforce' and 'High downforce' in actual_setup:
                suggestions[driver] = "Reduce downforce for better straight line speed"
            elif required == 'Medium downforce' and actual_setup != 'Balanced setup':
                suggestions[driver] = "Move towards a more balanced setup"
            else:
                suggestions[driver] = "Setup appears appropriate for this track"
        
        return suggestions
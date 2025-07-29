import pandas as pd
import numpy as np
from utils.data_loader import DataLoader

class TirePerformanceAnalyzer:
    """Analyze tire performance and degradation patterns"""
    
    def __init__(self):
        self.data_loader = DataLoader()
    
    def analyze_race_tire_performance(self, year, grand_prix):
        """Comprehensive tire performance analysis for a race"""
        try:
            session_data = self.data_loader.load_session_data(year, grand_prix, 'Race')
            if session_data is None:
                return None
            
            analysis = {
                'compound_performance': self.analyze_compound_performance(session_data),
                'degradation_analysis': self.analyze_tire_degradation(session_data),
                'temperature_impact': self.analyze_temperature_impact(session_data),
                'stint_optimization': self.analyze_stint_optimization(session_data),
                'tire_strategy_effectiveness': self.evaluate_tire_strategies(session_data)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_compound_performance(self, session_data):
        """Analyze performance of different tire compounds"""
        try:
            compound_data = {}
            
            # Get all available compounds used in the race
            all_laps = session_data.laps
            compounds = all_laps['Compound'].dropna().unique()
            
            for compound in compounds:
                compound_laps = all_laps[all_laps['Compound'] == compound]
                
                if compound_laps.empty:
                    continue
                
                # Calculate performance metrics
                lap_times = compound_laps['LapTime'].dropna()
                lap_times_seconds = [lt.total_seconds() for lt in lap_times]
                
                compound_data[compound] = {
                    'total_laps': len(compound_laps),
                    'average_lap_time': float(np.mean(lap_times_seconds)),
                    'best_lap_time': float(np.min(lap_times_seconds)),
                    'lap_time_std': float(np.std(lap_times_seconds)),
                    'drivers_used': len(compound_laps['Driver'].unique()),
                    'performance_window': self.calculate_performance_window(compound_laps),
                    'degradation_characteristics': self.analyze_compound_degradation(compound_laps)
                }
            
            # Rank compounds by performance
            compound_ranking = self.rank_compounds_by_performance(compound_data)
            
            return {
                'compound_data': compound_data,
                'compound_ranking': compound_ranking,
                'optimal_compound': compound_ranking[0] if compound_ranking else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_tire_degradation(self, session_data):
        """Analyze tire degradation patterns"""
        try:
            degradation_data = {}
            
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                driver_stints = self.extract_driver_stints(driver_laps)
                stint_analysis = []
                
                for stint in driver_stints:
                    if len(stint) >= 3:  # Need minimum laps for degradation analysis
                        degradation = self.calculate_stint_degradation(stint)
                        if degradation:
                            stint_analysis.append(degradation)
                
                if stint_analysis:
                    degradation_data[driver] = {
                        'stints': stint_analysis,
                        'average_degradation': float(np.mean([s['degradation_rate'] for s in stint_analysis])),
                        'degradation_consistency': float(np.std([s['degradation_rate'] for s in stint_analysis])),
                        'tire_management_rating': self.rate_tire_management(stint_analysis)
                    }
            
            return degradation_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_temperature_impact(self, session_data):
        """Analyze impact of temperature on tire performance"""
        try:
            # Get weather data if available
            weather_data = self.data_loader.get_weather_data(session_data)
            if weather_data is None or weather_data.empty:
                return {'error': 'No weather data available'}
            
            avg_air_temp = weather_data['AirTemp'].mean()
            avg_track_temp = weather_data['TrackTemp'].mean()
            
            temperature_impact = {
                'air_temperature': float(avg_air_temp),
                'track_temperature': float(avg_track_temp),
                'temperature_impact_rating': self.rate_temperature_impact(avg_track_temp),
                'compound_recommendations': self.recommend_compounds_for_temperature(avg_track_temp),
                'degradation_factor': self.calculate_temperature_degradation_factor(avg_track_temp)
            }
            
            return temperature_impact
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_stint_optimization(self, session_data):
        """Analyze optimal stint lengths for different compounds"""
        try:
            optimization_data = {}
            
            all_laps = session_data.laps
            compounds = all_laps['Compound'].dropna().unique()
            
            for compound in compounds:
                compound_stints = self.get_compound_stints(all_laps, compound)
                
                if not compound_stints:
                    continue
                
                stint_lengths = [len(stint) for stint in compound_stints]
                stint_performances = [self.calculate_stint_performance(stint) for stint in compound_stints]
                
                optimization_data[compound] = {
                    'average_stint_length': float(np.mean(stint_lengths)),
                    'optimal_stint_length': self.calculate_optimal_stint_length(compound_stints),
                    'stint_length_range': {
                        'min': int(np.min(stint_lengths)),
                        'max': int(np.max(stint_lengths))
                    },
                    'performance_correlation': self.correlate_length_with_performance(stint_lengths, stint_performances)
                }
            
            return optimization_data
            
        except Exception as e:
            return {'error': str(e)}
    
    def evaluate_tire_strategies(self, session_data):
        """Evaluate effectiveness of different tire strategies"""
        try:
            strategy_evaluation = {}
            
            # Get race results for strategy effectiveness
            results = session_data.results if hasattr(session_data, 'results') else None
            
            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                strategy = self.extract_tire_strategy(driver_laps)
                
                # Get final position if available
                final_position = None
                if results is not None:
                    driver_result = results[results['Abbreviation'] == driver]
                    if not driver_result.empty:
                        final_position = int(driver_result['Position'].iloc[0])
                
                strategy_evaluation[driver] = {
                    'strategy': strategy,
                    'final_position': final_position,
                    'strategy_effectiveness': self.rate_strategy_effectiveness(strategy, final_position),
                    'alternative_strategies': self.suggest_alternative_strategies(strategy),
                    'tire_usage_efficiency': self.calculate_tire_usage_efficiency(driver_laps)
                }
            
            return strategy_evaluation
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_performance_window(self, compound_laps):
        """Calculate the performance window for a tire compound"""
        try:
            # Group laps by tire age
            tire_ages = compound_laps['TyreLife'].dropna()
            if tire_ages.empty:
                return {'peak_performance_age': None, 'cliff_age': None}
            
            # Find peak performance age (lowest average lap time)
            age_performance = {}
            for age in tire_ages.unique():
                age_laps = compound_laps[compound_laps['TyreLife'] == age]
                lap_times = age_laps['LapTime'].dropna()
                if not lap_times.empty:
                    avg_time = np.mean([lt.total_seconds() for lt in lap_times])
                    age_performance[age] = avg_time
            
            if not age_performance:
                return {'peak_performance_age': None, 'cliff_age': None}
            
            peak_age = min(age_performance, key=age_performance.get)
            
            # Find cliff point (where performance degrades significantly)
            cliff_age = self.find_performance_cliff(age_performance)
            
            return {
                'peak_performance_age': int(peak_age),
                'cliff_age': cliff_age,
                'performance_curve': age_performance
            }
            
        except Exception as e:
            return {'peak_performance_age': None, 'cliff_age': None}
    
    # Helper methods with simplified implementations
    def rank_compounds_by_performance(self, compound_data):
        """Rank compounds by average performance"""
        try:
            rankings = []
            for compound, data in compound_data.items():
                if 'average_lap_time' in data and data['average_lap_time'] > 0:
                    rankings.append((compound, data['average_lap_time']))
            rankings.sort(key=lambda x: x[1])  # Sort by lap time (ascending)
            return [compound for compound, _ in rankings]
        except:
            return []
    
    def extract_driver_stints(self, driver_laps):
        """Extract individual stints from driver's laps"""
        try:
            stints = []
            current_stint = []
            current_compound = None
            
            for _, lap in driver_laps.iterrows():
                if lap['Compound'] != current_compound:
                    if current_stint:
                        stints.append(current_stint)
                    current_stint = [lap]
                    current_compound = lap['Compound']
                else:
                    current_stint.append(lap)
            
            if current_stint:
                stints.append(current_stint)
            
            return stints
        except:
            return []
    
    def calculate_stint_degradation(self, stint):
        """Calculate degradation metrics for a stint"""
        try:
            stint_df = pd.DataFrame(stint)
            lap_times = stint_df['LapTime'].dropna()
            if len(lap_times) < 3:
                return None
            
            lap_times_seconds = [lt.total_seconds() for lt in lap_times]
            
            # Calculate degradation rate (slope of lap time vs lap number)
            lap_numbers = list(range(len(lap_times_seconds)))
            degradation_rate = np.polyfit(lap_numbers, lap_times_seconds, 1)[0] if len(lap_times_seconds) > 1 else 0
            
            return {
                'compound': stint[0]['Compound'],
                'stint_length': len(stint),
                'degradation_rate': float(degradation_rate),
                'initial_performance': lap_times_seconds[0],
                'final_performance': lap_times_seconds[-1]
            }
        except:
            return None
    
    # Additional helper methods with simplified implementations
    def rate_tire_management(self, stint_analysis):
        """Rate tire management based on degradation consistency"""
        if not stint_analysis:
            return 0.0
        
        degradation_rates = [s['degradation_rate'] for s in stint_analysis]
        avg_degradation = np.mean(degradation_rates)
        
        # Lower degradation rate = better tire management
        if avg_degradation < 0.1:
            return 90.0
        elif avg_degradation < 0.2:
            return 75.0
        elif avg_degradation < 0.5:
            return 60.0
        else:
            return 40.0
    
    def rate_temperature_impact(self, track_temp):
        """Rate temperature impact on tire performance"""
        if track_temp < 30:
            return "Low impact - Cool conditions"
        elif track_temp < 40:
            return "Medium impact - Optimal conditions"
        elif track_temp < 50:
            return "High impact - Hot conditions"
        else:
            return "Very high impact - Extreme heat"
    
    def recommend_compounds_for_temperature(self, track_temp):
        """Recommend tire compounds based on temperature"""
        if track_temp < 30:
            return ["Soft", "Medium"]
        elif track_temp < 45:
            return ["Medium", "Hard"]
        else:
            return ["Hard", "Medium"]
    
    def calculate_temperature_degradation_factor(self, track_temp):
        """Calculate how temperature affects degradation"""
        base_temp = 35
        temp_diff = track_temp - base_temp
        return 1.0 + (temp_diff * 0.02)  # 2% increase per degree above base
    
    def get_compound_stints(self, all_laps, compound):
        """Get all stints for a specific compound"""
        compound_laps = all_laps[all_laps['Compound'] == compound]
        return self.extract_driver_stints(compound_laps)
    
    def calculate_stint_performance(self, stint):
        """Calculate overall performance of a stint"""
        try:
            stint_df = pd.DataFrame(stint)
            lap_times = stint_df['LapTime'].dropna()
            if lap_times.empty:
                return 0
            
            lap_times_seconds = [lt.total_seconds() for lt in lap_times]
            return np.mean(lap_times_seconds)
        except:
            return 0
    
    def calculate_optimal_stint_length(self, compound_stints):
        """Calculate optimal stint length for a compound"""
        try:
            stint_lengths = [len(stint) for stint in compound_stints]
            return int(np.mean(stint_lengths)) if stint_lengths else 0
        except:
            return 0
    
    def correlate_length_with_performance(self, lengths, performances):
        """Correlate stint length with performance"""
        try:
            if len(lengths) == len(performances) and len(lengths) > 1:
                correlation = np.corrcoef(lengths, performances)[0, 1]
                return float(correlation)
            return 0.0
        except:
            return 0.0
    
    def extract_tire_strategy(self, driver_laps):
        """Extract tire strategy from driver laps"""
        try:
            strategy = {
                'total_stops': 0,
                'compounds_used': [],
                'stint_lengths': []
            }
            
            compounds = driver_laps['Compound'].dropna().unique()
            strategy['compounds_used'] = list(compounds)
            strategy['total_stops'] = len(compounds) - 1  # Subtract 1 because first compound doesn't require a stop
            
            return strategy
        except:
            return {'total_stops': 0, 'compounds_used': [], 'stint_lengths': []}
    
    def rate_strategy_effectiveness(self, strategy, final_position):
        """Rate strategy effectiveness"""
        if final_position is None:
            return 50.0
        
        # Simple rating based on final position and number of stops
        base_score = max(0, 110 - (final_position * 5))  # Higher positions get higher scores
        
        # Adjust for number of stops
        if strategy['total_stops'] == 1:
            return min(100.0, base_score + 10)  # Bonus for one-stop
        elif strategy['total_stops'] > 3:
            return max(0.0, base_score - 20)  # Penalty for many stops
        
        return min(100.0, base_score)
    
    def suggest_alternative_strategies(self, strategy):
        """Suggest alternative tire strategies"""
        alternatives = []
        
        if strategy['total_stops'] == 0:
            alternatives.append("Consider a one-stop strategy for better track position")
        elif strategy['total_stops'] == 1:
            alternatives.append("Consider a two-stop for fresher tires")
        elif strategy['total_stops'] >= 2:
            alternatives.append("Consider fewer stops to reduce time loss")
        
        return alternatives
    
    def calculate_tire_usage_efficiency(self, driver_laps):
        """Calculate how efficiently tires were used"""
        try:
            valid_laps = driver_laps['LapTime'].dropna()
            if len(valid_laps) < 5:
                return 50.0
            
            lap_times_seconds = [lt.total_seconds() for lt in valid_laps]
            cv = np.std(lap_times_seconds) / np.mean(lap_times_seconds)
            
            # Lower coefficient of variation = better efficiency
            efficiency = max(0, 100 * (1 - cv * 10))
            return min(100.0, efficiency)
        except:
            return 50.0
    
    def find_performance_cliff(self, age_performance):
        """Find where tire performance drops significantly"""
        try:
            ages = sorted(age_performance.keys())
            if len(ages) < 3:
                return None
            
            # Look for significant performance drop
            for i in range(1, len(ages)):
                current_time = age_performance[ages[i]]
                prev_time = age_performance[ages[i-1]]
                
                # If lap time increases by more than 2%, consider it a cliff
                if (current_time - prev_time) / prev_time > 0.02:
                    return int(ages[i])
            
            return None
        except:
            return None
    
    def analyze_compound_degradation(self, compound_laps):
        """Analyze degradation characteristics of a compound"""
        try:
            # Group by stint and calculate degradation for each
            stints = self.group_laps_by_stint(compound_laps)
            degradation_rates = []
            
            for stint in stints:
                if len(stint) >= 3:
                    rate = self.calculate_stint_degradation_rate(stint)
                    if rate is not None:
                        degradation_rates.append(rate)
            
            if not degradation_rates:
                return {'average_degradation_rate': None, 'degradation_consistency': None}
            
            return {
                'average_degradation_rate': float(np.mean(degradation_rates)),
                'degradation_consistency': float(np.std(degradation_rates)),
                'degradation_range': {
                    'min': float(np.min(degradation_rates)),
                    'max': float(np.max(degradation_rates))
                }
            }
        except:
            return {'average_degradation_rate': None, 'degradation_consistency': None}
    
    def group_laps_by_stint(self, laps):
        """Group laps by stint (simplified)"""
        # This is a simplified grouping - in reality would need more complex logic
        return [laps.iloc[i:i+10] for i in range(0, len(laps), 10)]
    
    def calculate_stint_degradation_rate(self, stint):
        """Calculate degradation rate for a stint"""
        try:
            lap_times = stint['LapTime'].dropna()
            if len(lap_times) < 3:
                return None
            
            lap_times_seconds = [lt.total_seconds() for lt in lap_times]
            lap_numbers = list(range(len(lap_times_seconds)))
            
            # Calculate slope (degradation rate)
            slope = np.polyfit(lap_numbers, lap_times_seconds, 1)[0] if len(lap_times_seconds) > 1 else 0
            return float(slope)
        except:
            return None
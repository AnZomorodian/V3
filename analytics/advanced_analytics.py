import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from utils.json_utils import make_json_serializable, format_lap_time
from typing import Dict, Any
from datetime import datetime
import logging

class AdvancedF1Analytics:
    """Advanced F1 analytics and insights"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a handler that writes log messages to a file
        handler = logging.FileHandler('f1_analytics.log')
        handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)


    def comprehensive_session_analysis(self, year, grand_prix, session):
        """Comprehensive analysis of a session"""
        try:
            session_data = self.data_loader.load_session_data(year, grand_prix, session)
            if session_data is None:
                return None

            analysis = {
                'performance_analysis': self.analyze_performance_metrics(session_data),
                'consistency_analysis': self.analyze_consistency(session_data),
                'tire_degradation': self.analyze_tire_degradation(session_data),
                'sector_analysis': self.analyze_sector_performance(session_data),
                'race_pace': self.analyze_race_pace(session_data) if session == 'Race' else None
            }

            # Ensure all data is JSON serializable
            return make_json_serializable(analysis)

        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {
                'error': str(e),
                'analysis_type': 'comprehensive_session_analysis',
                'timestamp': datetime.now().isoformat()
            }

    def analyze_performance_metrics(self, session_data):
        """Analyze driver performance metrics"""
        try:
            performance_data = {}

            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_drivers(driver)
                if driver_laps.empty:
                    continue

                try:
                    fastest_lap = driver_laps.pick_fastest()

                    # Safely extract data with fallbacks
                    fastest_time = fastest_lap['LapTime'] if pd.notna(fastest_lap['LapTime']) else None
                    avg_time = driver_laps['LapTime'].mean() if not driver_laps['LapTime'].isna().all() else None
                    position = fastest_lap['Position'] if pd.notna(fastest_lap['Position']) else None

                    performance_data[str(driver)] = {
                        'fastest_lap_time': format_lap_time(fastest_time),
                        'average_lap_time': format_lap_time(avg_time),
                        'total_laps': int(len(driver_laps)),
                        'valid_laps': int(len(driver_laps.dropna(subset=['LapTime']))),
                        'position': int(position) if position is not None else None
                    }

                    # Add telemetry insights if available
                    try:
                        telemetry = fastest_lap.get_telemetry()
                        if not telemetry.empty and 'Speed' in telemetry.columns:
                            speed_data = telemetry['Speed'].dropna()
                            rpm_data = telemetry['RPM'].dropna() if 'RPM' in telemetry.columns else pd.Series()
                            throttle_data = telemetry['Throttle'].dropna() if 'Throttle' in telemetry.columns else pd.Series()

                            performance_data[str(driver)].update({
                                'max_speed': float(speed_data.max()) if not speed_data.empty else None,
                                'avg_speed': float(speed_data.mean()) if not speed_data.empty else None,
                                'max_rpm': float(rpm_data.max()) if not rpm_data.empty else None,
                                'avg_throttle': float(throttle_data.mean()) if not throttle_data.empty else None
                            })
                    except Exception:
                        # Telemetry not available or failed to load
                        pass

                except Exception:
                    # Skip this driver if data is corrupted
                    continue

            return performance_data

        except Exception as e:
            return {'error': f'Performance analysis failed: {str(e)}'}

    def analyze_consistency(self, session_data):
        """Analyze driver consistency"""
        try:
            consistency_data = {}

            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_drivers(driver)
                if len(driver_laps) < 3:  # Need minimum laps for consistency analysis
                    continue

                lap_times = driver_laps['LapTime'].dropna()
                if len(lap_times) < 3:
                    continue

                # Convert to seconds for calculation
                try:
                    lap_times_seconds = [lt.total_seconds() for lt in lap_times if pd.notna(lt)]

                    if len(lap_times_seconds) < 3:
                        continue

                    std_dev = np.std(lap_times_seconds)
                    mean_time = np.mean(lap_times_seconds)
                    cv = std_dev / mean_time if mean_time > 0 else 0

                    consistency_data[str(driver)] = {
                        'standard_deviation': float(std_dev),
                        'coefficient_of_variation': float(cv),
                        'consistency_score': float(self.calculate_consistency_score(lap_times_seconds)),
                        'outlier_laps': self.identify_outlier_laps(lap_times_seconds),
                        'consistency_windows': self.analyze_consistency_windows(lap_times_seconds)
                    }
                except Exception:
                    continue

            return consistency_data

        except Exception as e:
            return {'error': f'Consistency analysis failed: {str(e)}'}

    def analyze_tire_degradation(self, session_data):
        """Analyze tire degradation patterns"""
        try:
            degradation_data = {}

            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_drivers(driver)
                if driver_laps.empty:
                    continue

                # Group by stint (tire compound changes)
                stints = []
                current_compound = None
                current_stint = []

                for _, lap in driver_laps.iterrows():
                    compound = lap.get('Compound', 'UNKNOWN')
                    if compound != current_compound:
                        if current_stint:
                            stints.append(current_stint)
                        current_stint = [lap]
                        current_compound = compound
                    else:
                        current_stint.append(lap)

                if current_stint:
                    stints.append(current_stint)

                # Analyze each stint
                stint_analysis = []
                for i, stint in enumerate(stints):
                    if len(stint) > 2:  # Need minimum laps for degradation analysis
                        try:
                            stint_df = pd.DataFrame(stint)
                            degradation_rate = self.calculate_tire_degradation(stint_df)

                            start_time = stint[0].get('LapTime')
                            end_time = stint[-1].get('LapTime')

                            stint_analysis.append({
                                'stint_number': i + 1,
                                'compound': str(stint[0].get('Compound', 'UNKNOWN')),
                                'laps': len(stint),
                                'degradation_rate': float(degradation_rate),
                                'start_performance': str(start_time) if pd.notna(start_time) else None,
                                'end_performance': str(end_time) if pd.notna(end_time) else None
                            })
                        except Exception:
                            continue

                degradation_data[str(driver)] = stint_analysis

            return degradation_data

        except Exception as e:
            return {'error': f'Tire degradation analysis failed: {str(e)}'}

    def analyze_sector_performance(self, session_data):
        """Analyze sector-by-sector performance"""
        try:
            sector_data = {}

            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_drivers(driver)
                if driver_laps.empty:
                    continue

                # Check which sector columns exist
                sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
                available_sectors = [col for col in sector_cols if col in driver_laps.columns]

                if not available_sectors:
                    continue

                valid_laps = driver_laps.dropna(subset=available_sectors)
                if valid_laps.empty:
                    continue

                sector_info = {}

                for i, sector_col in enumerate(available_sectors, 1):
                    sector_times = valid_laps[sector_col].dropna()
                    if not sector_times.empty:
                        try:
                            best_time = sector_times.min()
                            avg_time = sector_times.mean()
                            std_time = sector_times.std()

                            sector_info[f'best_sector_{i}'] = str(best_time) if pd.notna(best_time) else None
                            sector_info[f'avg_sector_{i}'] = str(avg_time) if pd.notna(avg_time) else None
                            sector_info[f'sector_{i}_std'] = float(std_time.total_seconds()) if pd.notna(std_time) else None
                        except Exception:
                            continue

                if sector_info:
                    sector_data[str(driver)] = sector_info

            return sector_data

        except Exception as e:
            return {'error': f'Sector analysis failed: {str(e)}'}

    def analyze_race_pace(self, session_data):
        """Analyze race pace and strategy"""
        try:
            if not hasattr(session_data, 'laps'):
                return {'error': 'No lap data available'}

            race_data = {}

            for driver in session_data.drivers:
                driver_laps = session_data.laps.pick_drivers(driver)
                if driver_laps.empty:
                    continue

                # Filter out outlier laps (pit stops, safety cars, etc.)
                clean_laps = self.filter_clean_laps(driver_laps)

                if len(clean_laps) < 5:
                    continue

                try:
                    lap_times = clean_laps['LapTime'].dropna()

                    if not lap_times.empty:
                        avg_pace = lap_times.mean()
                        median_pace = lap_times.median()
                        fastest_lap = lap_times.min()
                        pace_std = lap_times.std()

                        # Convert lap times to seconds for pace analysis
                        lap_times_seconds = [lt.total_seconds() for lt in lap_times if pd.notna(lt)]
                        representative_pace = self.calculate_race_pace(lap_times_seconds)

                        race_data[str(driver)] = {
                            'average_pace': str(avg_pace) if pd.notna(avg_pace) else None,
                            'median_pace': str(median_pace) if pd.notna(median_pace) else None,
                            'pace_consistency': float(pace_std.total_seconds()) if pd.notna(pace_std) else None,
                            'fastest_race_lap': str(fastest_lap) if pd.notna(fastest_lap) else None,
                            'stint_analysis': self.analyze_race_stints(driver_laps),
                            'position_changes': self.analyze_position_changes(driver_laps),
                            'representative_pace': float(representative_pace) if representative_pace is not None else None
                        }
                except Exception:
                    continue

            return race_data

        except Exception as e:
            return {'error': f'Race pace analysis failed: {str(e)}'}

    def calculate_consistency_score(self, lap_times):
        """Calculate a consistency score (0-100, higher is more consistent)"""
        try:
            if len(lap_times) < 2:
                return 0.0

            cv = np.std(lap_times) / np.mean(lap_times)
            # Convert to 0-100 scale (lower CV = higher consistency)
            consistency_score = max(0.0, 100.0 - (cv * 1000))
            return consistency_score

        except Exception:
            return 0.0

    def identify_outlier_laps(self, lap_times):
        """Identify outlier laps using statistical methods"""
        try:
            if len(lap_times) < 4:
                return []

            q1 = np.percentile(lap_times, 25)
            q3 = np.percentile(lap_times, 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = []
            for i, lap_time in enumerate(lap_times):
                if lap_time < lower_bound or lap_time > upper_bound:
                    outliers.append(i + 1)  # 1-indexed lap numbers

            return outliers

        except Exception:
            return []

    def calculate_tire_degradation(self, stint_data):
        """Calculate tire degradation rate for a stint"""
        try:
            if len(stint_data) < 3:
                return 0.0

            lap_times = []
            for _, row in stint_data.iterrows():
                lap_time = row.get('LapTime')
                if pd.notna(lap_time):
                    lap_times.append(lap_time.total_seconds())

            if len(lap_times) < 3:
                return 0.0

            lap_numbers = list(range(1, len(lap_times) + 1))

            # Linear regression to find degradation trend
            slope = np.polyfit(lap_numbers, lap_times, 1)[0]
            return slope  # seconds per lap degradation

        except Exception:
            return 0.0

    def filter_clean_laps(self, driver_laps):
        """Filter out non-representative laps (pit stops, etc.)"""
        try:
            # Remove laps with pit stops if columns exist
            clean_laps = driver_laps.copy()

            if 'PitOutTime' in clean_laps.columns and 'PitInTime' in clean_laps.columns:
                clean_laps = clean_laps[clean_laps['PitOutTime'].isna() & clean_laps['PitInTime'].isna()]

            # Remove statistical outliers if we have enough data
            if len(clean_laps) > 5 and 'LapTime' in clean_laps.columns:
                lap_times_data = clean_laps['LapTime'].dropna()
                if not lap_times_data.empty:
                    lap_times = [lt.total_seconds() for lt in lap_times_data]
                    q1 = np.percentile(lap_times, 25)
                    q3 = np.percentile(lap_times, 75)
                    iqr = q3 - q1

                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    clean_laps = clean_laps[
                        clean_laps['LapTime'].apply(lambda x: 
                            pd.isna(x) or (lower_bound <= x.total_seconds() <= upper_bound)
                        )
                    ]

            return clean_laps

        except Exception:
            return driver_laps

    def analyze_race_stints(self, driver_laps):
        """Analyze individual race stints"""
        try:
            stints = []
            current_compound = None
            current_stint_laps = []

            for _, lap in driver_laps.iterrows():
                compound = lap.get('Compound', 'UNKNOWN')
                if compound != current_compound:
                    if current_stint_laps:
                        stint_analysis = self.analyze_single_stint(current_stint_laps, current_compound)
                        if stint_analysis:
                            stints.append(stint_analysis)

                    current_stint_laps = [lap]
                    current_compound = compound
                else:
                    current_stint_laps.append(lap)

            # Analyze final stint
            if current_stint_laps:
                stint_analysis = self.analyze_single_stint(current_stint_laps, current_compound)
                if stint_analysis:
                    stints.append(stint_analysis)

            return stints

        except Exception:
            return []

    def analyze_single_stint(self, stint_laps, compound):
        """Analyze a single stint"""
        try:
            stint_df = pd.DataFrame(stint_laps)

            if 'LapTime' not in stint_df.columns or stint_df['LapTime'].isna().all():
                return None

            lap_times = stint_df['LapTime'].dropna()
            if lap_times.empty:
                return None

            avg_time = lap_times.mean()
            best_time = lap_times.min()
            degradation = self.calculate_tire_degradation(stint_df)

            lap_numbers = stint_df['LapNumber'].dropna() if 'LapNumber' in stint_df.columns else pd.Series()

            return {
                'compound': str(compound) if compound else 'UNKNOWN',
                'laps': len(stint_laps),
                'avg_lap_time': str(avg_time) if pd.notna(avg_time) else None,
                'best_lap_time': str(best_time) if pd.notna(best_time) else None,
                'degradation': float(degradation),
                'start_lap': int(lap_numbers.min()) if not lap_numbers.empty else None,
                'end_lap': int(lap_numbers.max()) if not lap_numbers.empty else None
            }

        except Exception:
            return None

    def analyze_position_changes(self, driver_laps):
        """Analyze position changes throughout the race"""
        try:
            if 'Position' not in driver_laps.columns:
                return {'error': 'Position data not available'}

            positions = driver_laps['Position'].dropna()
            if positions.empty:
                return {'error': 'No valid position data'}

            start_position = int(positions.iloc[0])
            end_position = int(positions.iloc[-1])
            highest_position = int(positions.min())
            lowest_position = int(positions.max())
            position_variance = float(positions.var()) if len(positions) > 1 else 0.0

            return {
                'start_position': start_position,
                'end_position': end_position,
                'position_change': start_position - end_position,
                'highest_position': highest_position,
                'lowest_position': lowest_position,
                'position_variance': position_variance
            }

        except Exception:
            return {'error': 'Position analysis failed'}

    def analyze_weather_impact(self, year: int, grand_prix: str, session: str = 'Race') -> Dict[str, Any]:
        """Analyze weather impact on session performance"""
        try:
            self.logger.info(f"Running weather impact analysis for {year} {grand_prix} {session}")

            session_data = self.data_loader.load_session_data(year, grand_prix, session)
            if session_data is None:
                return {
                    'error': 'Failed to load session data',
                    'analysis_type': 'weather_impact',
                    'timestamp': datetime.now().isoformat()
                }

            # Get weather data
            weather_data = session_data.weather_data
            if weather_data.empty:
                return {
                    'error': 'No weather data available',
                    'analysis_type': 'weather_impact',
                    'timestamp': datetime.now().isoformat()
                }

            # Calculate weather statistics
            weather_conditions = {
                'avg_air_temp': round(weather_data['AirTemp'].mean(), 1),
                'max_air_temp': round(weather_data['AirTemp'].max(), 1),
                'min_air_temp': round(weather_data['AirTemp'].min(), 1),
                'avg_track_temp': round(weather_data['TrackTemp'].mean(), 1),
                'max_track_temp': round(weather_data['TrackTemp'].max(), 1),
                'min_track_temp': round(weather_data['TrackTemp'].min(), 1),
                'avg_humidity': round(weather_data['Humidity'].mean(), 1),
                'avg_pressure': round(weather_data['Pressure'].mean(), 2),
                'wind_speed': round(weather_data['WindSpeed'].mean(), 1) if 'WindSpeed' in weather_data.columns else 0
            }

            # Analyze impact on lap times
            laps = session_data.laps
            if not laps.empty:
                # Get valid lap times
                valid_laps = laps[laps['LapTime'].notna()]
                if not valid_laps.empty:
                    # Convert lap times to seconds for analysis
                    lap_times_seconds = valid_laps['LapTime'].dt.total_seconds()

                    weather_impact = {
                        'temperature_correlation': {
                            'air_temp_vs_laptime': np.corrcoef(
                                weather_data['AirTemp'][:len(lap_times_seconds)], 
                                lap_times_seconds[:len(weather_data['AirTemp'])]
                            )[0,1] if len(weather_data['AirTemp']) > 1 and len(lap_times_seconds) > 1 else 0,
                            'track_temp_vs_laptime': np.corrcoef(
                                weather_data['TrackTemp'][:len(lap_times_seconds)], 
                                lap_times_seconds[:len(weather_data['TrackTemp'])]
                            )[0,1] if len(weather_data['TrackTemp']) > 1 and len(lap_times_seconds) > 1 else 0
                        },
                        'optimal_conditions': {
                            'air_temp_range': f"{weather_conditions['min_air_temp']}-{weather_conditions['max_air_temp']}°C",
                            'track_temp_range': f"{weather_conditions['min_track_temp']}-{weather_conditions['max_track_temp']}°C",
                            'humidity_level': f"{weather_conditions['avg_humidity']}%"
                        }
                    }
                else:
                    weather_impact = {'error': 'No valid lap times for correlation analysis'}
            else:
                weather_impact = {'error': 'No lap data available'}

            return make_json_serializable({
                'analysis_type': 'weather_impact',
                'session_info': {
                    'year': year,
                    'grand_prix': grand_prix,
                    'session': session
                },
                'weather_conditions': weather_conditions,
                'weather_impact': weather_impact,
                'data_points': len(weather_data),
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"Error in weather analysis: {str(e)}")
            return {
                'error': str(e),
                'analysis_type': 'weather_impact',
                'timestamp': datetime.now().isoformat()
            }

    def compare_drivers(self, year: int, grand_prix: str, session: str = 'Race') -> Dict[str, Any]:
        """Compare driver performance head-to-head with timeout protection"""
        try:
            self.logger.info(f"Running driver comparison for {year} {grand_prix} {session}")

            # Try to load session data with timeout protection
            try:
                session_data = self.data_loader.load_session_data(year, grand_prix, session)
            except Exception as load_error:
                self.logger.error(f"Data loading failed: {str(load_error)}")
                return {
                    'error': f'Failed to load session data: {str(load_error)}',
                    'analysis_type': 'driver_comparison',
                    'timestamp': datetime.now().isoformat()
                }

            if session_data is None:
                return {
                    'error': 'No session data available',
                    'analysis_type': 'driver_comparison',
                    'timestamp': datetime.now().isoformat()
                }

            # Check if laps data exists
            if not hasattr(session_data, 'laps') or session_data.laps.empty:
                return {
                    'error': 'No lap data available for this session',
                    'analysis_type': 'driver_comparison',
                    'timestamp': datetime.now().isoformat()
                }

            laps = session_data.laps

            # Get all drivers with valid data
            drivers = laps['Driver'].unique()[:10]  # Limit to 10 drivers to prevent timeout
            comparison_data = []

            for driver in drivers:
                try:
                    driver_laps = laps[laps['Driver'] == driver]
                    valid_laps = driver_laps[driver_laps['LapTime'].notna()]

                    if not valid_laps.empty:
                        # Calculate basic statistics without heavy telemetry processing
                        lap_times = valid_laps['LapTime'].dt.total_seconds()
                        fastest_lap_idx = valid_laps['LapTime'].idxmin()
                        fastest_lap = valid_laps.loc[fastest_lap_idx]

                        # Get position data safely
                        try:
                            if 'Position' in driver_laps.columns:
                                final_position = driver_laps['Position'].dropna().iloc[-1] if not driver_laps['Position'].dropna().empty else 'N/A'
                            else:
                                final_position = 'N/A'
                        except:
                            final_position = 'N/A'

                        # Basic telemetry without timeout risk
                        top_speed = 0
                        try:
                            # Only get telemetry for fastest lap with timeout protection
                            if hasattr(fastest_lap, 'get_telemetry'):
                                telemetry = fastest_lap.get_telemetry()
                                if not telemetry.empty and 'Speed' in telemetry.columns:
                                    speed_data = telemetry['Speed'].dropna()
                                    if not speed_data.empty:
                                        top_speed = float(speed_data.max())
                        except Exception as tel_error:
                            self.logger.warning(f"Telemetry unavailable for {driver}: {str(tel_error)}")
                            top_speed = 0

                        driver_stats = {
                            'driver': str(driver),
                            'best_lap': format_lap_time(fastest_lap['LapTime']),
                            'avg_lap': format_lap_time(pd.Timedelta(seconds=lap_times.mean())) if len(lap_times) > 0 else None,
                            'lap_count': int(len(valid_laps)),
                            'top_speed': round(top_speed, 1) if top_speed > 0 else 'N/A',
                            'position': int(final_position) if final_position != 'N/A' and pd.notna(final_position) else 'N/A',
                            'consistency': round(lap_times.std(), 3) if len(lap_times) > 1 else 0.0
                        }

                        comparison_data.append(driver_stats)

                except Exception as driver_error:
                    self.logger.warning(f"Skipping driver {driver}: {str(driver_error)}")
                    continue

            # Sort by best lap time
            def sort_key(x):
                if x['best_lap'] and x['best_lap'] != 'N/A':
                    try:
                        # Convert formatted time back to seconds for sorting
                        time_parts = x['best_lap'].split(':')
                        if len(time_parts) == 2:
                            return float(time_parts[0]) * 60 + float(time_parts[1])
                    except:
                        pass
                return 999999

            comparison_data.sort(key=sort_key)

            # Calculate gaps to fastest
            if comparison_data and comparison_data[0]['best_lap'] != 'N/A':
                fastest_time_str = comparison_data[0]['best_lap']
                try:
                    fastest_seconds = sort_key(comparison_data[0])
                    
                    for driver_data in comparison_data:
                        if driver_data['best_lap'] != 'N/A':
                            driver_seconds = sort_key(driver_data)
                            gap_seconds = driver_seconds - fastest_seconds
                            if gap_seconds > 0:
                                driver_data['gap_to_fastest'] = f"+{gap_seconds:.3f}s"
                            else:
                                driver_data['gap_to_fastest'] = "Fastest"
                        else:
                            driver_data['gap_to_fastest'] = 'N/A'
                except:
                    for driver_data in comparison_data:
                        driver_data['gap_to_fastest'] = 'N/A'

            return make_json_serializable({
                'analysis_type': 'driver_comparison',
                'session_info': {
                    'year': year,
                    'grand_prix': grand_prix,
                    'session': session
                },
                'comparison_data': comparison_data,
                'total_drivers': len(comparison_data),
                'data_quality': 'Limited telemetry for performance',
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"Critical error in driver comparison: {str(e)}")
            return {
                'error': f'Driver comparison failed: {str(e)}',
                'analysis_type': 'driver_comparison',
                'suggestion': 'Try a different session or check data availability',
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_dimensional_advantage(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate advantage in hyperdimensional space"""
        if not dimension_scores:
            return 0.0

        # Calculate overall performance vector magnitude
        scores = list(dimension_scores.values())
        magnitude = np.linalg.norm(scores)

        # Normalize to 0-1 range
        return min(1.0, magnitude / len(scores))

    def analyze_consistency_windows(self, lap_times_seconds):
        """Analyze consistency in different time windows"""
        try:
            if len(lap_times_seconds) < 5:
                return {'error': 'Insufficient data for window analysis'}

            window_size = 5
            windows = []

            for i in range(0, len(lap_times_seconds) - window_size + 1, window_size):
                window = lap_times_seconds[i:i + window_size]
                if len(window) == window_size:
                    window_cv = np.std(window) / np.mean(window)
                    windows.append({
                        'laps': f"{i+1}-{i+window_size}",
                        'consistency_cv': float(window_cv),
                        'avg_time': float(np.mean(window)),
                        'best_time': float(min(window)),
                        'worst_time': float(max(window))
                    })

            if windows:
                best_window = min(windows, key=lambda x: x['consistency_cv'])
                worst_window = max(windows, key=lambda x: x['consistency_cv'])

                return {
                    'windows': windows,
                    'best_consistency_window': best_window,
                    'worst_consistency_window': worst_window,
                    'window_count': len(windows)
                }

            return {'error': 'No complete windows found'}

        except Exception as e:
            return {'error': str(e)}

    def calculate_race_pace(self, lap_times_seconds):
        """Calculate representative race pace"""
        try:
            if len(lap_times_seconds) < 5:
                return None

            # Remove fastest and slowest 10% to get representative pace
            sorted_times = sorted(lap_times_seconds)
            start_idx = int(len(sorted_times) * 0.1)
            end_idx = int(len(sorted_times) * 0.9)

            representative_times = sorted_times[start_idx:end_idx]
            return np.mean(representative_times) if representative_times else None

        except Exception:
            return None

    def analyze_speed_zones(self, speed_data):
        """Analyze speed zones and acceleration events"""
        try:
            if speed_data.empty:
                return {'high_speed_percentage': 0, 'acceleration_events': 0}

            max_speed = speed_data.max()
            high_speed_threshold = max_speed * 0.9

            high_speed_percentage = (speed_data >= high_speed_threshold).sum() / len(speed_data) * 100

            # Count acceleration events (speed increases > 20 km/h over 3 data points)
            acceleration_events = 0
            for i in range(3, len(speed_data)):
                speed_increase = speed_data.iloc[i] - speed_data.iloc[i-3]
                if speed_increase > 20:
                    acceleration_events += 1

            return {
                'high_speed_percentage': float(high_speed_percentage),
                'acceleration_events': acceleration_events
            }

        except Exception:
            return {'high_speed_percentage': 0, 'acceleration_events': 0}

    def analyze_throttle_usage(self, throttle_data):
        """Analyze throttle usage patterns"""
        try:
            if throttle_data.empty:
                return {'full_throttle_time': 0, 'smoothness_score': 0, 'aggressive_count': 0}

            # Full throttle percentage (>95%)
            full_throttle_time = (throttle_data > 95).sum() / len(throttle_data) * 100

            # Smoothness score (inverse of throttle variance)
            throttle_variance = throttle_data.var()
            smoothness_score = 100 / (1 + throttle_variance / 100)

            # Aggressive inputs (rapid changes > 50% in one data point)
            aggressive_count = 0
            for i in range(1, len(throttle_data)):
                change = abs(throttle_data.iloc[i] - throttle_data.iloc[i-1])
                if change > 50:
                    aggressive_count += 1

            return {
                'full_throttle_time': float(full_throttle_time),
                'smoothness_score': float(smoothness_score),
                'aggressive_count': aggressive_count
            }

        except Exception:
            return {'full_throttle_time': 0, 'smoothness_score': 0, 'aggressive_count': 0}

    def analyze_driver_sectors(self, driver_laps):
        """Analyze sector performance for individual driver"""
        try:
            sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
            available_sectors = [col for col in sector_cols if col in driver_laps.columns]

            if not available_sectors:
                return None

            sector_analysis = {}

            for i, sector_col in enumerate(available_sectors, 1):
                sector_times = driver_laps[sector_col].dropna()
                if not sector_times.empty:
                    sector_seconds = [st.total_seconds() for st in sector_times if pd.notna(st)]
                    if sector_seconds:
                        sector_analysis[f'sector_{i}'] = {
                            'best_time': float(min(sector_seconds)),
                            'avg_time': float(np.mean(sector_seconds)),
                            'worst_time': float(max(sector_seconds)),
                            'consistency': float(np.std(sector_seconds) / np.mean(sector_seconds)),
                            'improvement': float((sector_seconds[0] - sector_seconds[-1]) / sector_seconds[0] * 100) if len(sector_seconds) > 1 else 0
                        }

            return sector_analysis if sector_analysis else None

        except Exception:
            return None
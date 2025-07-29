
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from utils.json_utils import make_json_serializable

class AdvancedF1Analytics:
    """Advanced F1 analytics and insights"""

    def __init__(self):
        self.data_loader = DataLoader()

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
            return {'error': f'Analysis failed: {str(e)}'}

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
                        'fastest_lap_time': str(fastest_time) if fastest_time is not None else None,
                        'average_lap_time': str(avg_time) if avg_time is not None else None,
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
                        'outlier_laps': self.identify_outlier_laps(lap_times_seconds)
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
                        
                        race_data[str(driver)] = {
                            'average_pace': str(avg_pace) if pd.notna(avg_pace) else None,
                            'median_pace': str(median_pace) if pd.notna(median_pace) else None,
                            'pace_consistency': float(pace_std.total_seconds()) if pd.notna(pace_std) else None,
                            'fastest_race_lap': str(fastest_lap) if pd.notna(fastest_lap) else None,
                            'stint_analysis': self.analyze_race_stints(driver_laps),
                            'position_changes': self.analyze_position_changes(driver_laps)
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

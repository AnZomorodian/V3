"""
Driver Comparison Module
Advanced driver-to-driver comparison and analysis
"""

import fastf1
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import scipy.stats as stats

class DriverComparisonAnalyzer:
    """Advanced driver comparison and performance analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_comprehensive_comparison(self, year: int, grand_prix: str, session: str, 
                                      drivers: List[str]) -> Dict[str, Any]:
        """Create comprehensive comparison between multiple drivers"""
        return self.get_comprehensive_driver_comparison(year, grand_prix, session, drivers)
    
    def get_comprehensive_driver_comparison(self, year: int, grand_prix: str, session: str, 
                                          drivers: List[str]) -> Dict[str, Any]:
        """Comprehensive comparison between multiple drivers"""
        try:
            session_obj = fastf1.get_session(year, grand_prix, session)
            session_obj.load()
            
            comparison_data = {
                'driver_statistics': {},
                'performance_metrics': {},
                'telemetry_comparison': {},
                'sector_analysis': {},
                'consistency_metrics': {}
            }
            
            for driver in drivers:
                try:
                    driver_laps = session_obj.laps.pick_drivers(driver)
                    if not driver_laps.empty:
                        # Basic statistics
                        comparison_data['driver_statistics'][driver] = self._get_driver_statistics(driver_laps)
                        
                        # Performance metrics
                        comparison_data['performance_metrics'][driver] = self._calculate_performance_metrics(driver_laps)
                        
                        # Sector analysis
                        comparison_data['sector_analysis'][driver] = self._analyze_driver_sectors(driver_laps)
                        
                        # Consistency metrics
                        comparison_data['consistency_metrics'][driver] = self._calculate_consistency_metrics(driver_laps)
                        
                        # Telemetry comparison (fastest lap)
                        fastest_lap = driver_laps.pick_fastest()
                        if fastest_lap is not None:
                            telemetry = fastest_lap.get_telemetry()
                            comparison_data['telemetry_comparison'][driver] = self._extract_telemetry_metrics(telemetry)
                
                except Exception as driver_error:
                    self.logger.warning(f"Error analyzing driver {driver}: {str(driver_error)}")
                    continue
            
            # Add comparative analysis
            comparison_data['comparative_analysis'] = self._generate_comparative_insights(comparison_data, drivers)
            
            return {
                'driver_comparison': comparison_data,
                'session_info': {
                    'year': year,
                    'grand_prix': grand_prix,
                    'session': session,
                    'drivers_compared': drivers
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive driver comparison: {str(e)}")
            return {'error': str(e)}
    
    def _get_driver_statistics(self, driver_laps: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic driver statistics"""
        valid_laps = driver_laps['LapTime'].dropna()
        
        if valid_laps.empty:
            return {'total_laps': 0, 'valid_laps': 0}
        
        lap_times_seconds = [lt.total_seconds() for lt in valid_laps]
        
        return {
            'total_laps': len(driver_laps),
            'valid_laps': len(valid_laps),
            'fastest_lap_time': float(min(lap_times_seconds)),
            'average_lap_time': float(np.mean(lap_times_seconds)),
            'slowest_lap_time': float(max(lap_times_seconds)),
            'lap_time_range': float(max(lap_times_seconds) - min(lap_times_seconds)),
            'median_lap_time': float(np.median(lap_times_seconds))
        }
    
    def _calculate_performance_metrics(self, driver_laps: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        valid_laps = driver_laps['LapTime'].dropna()
        
        if len(valid_laps) < 2:
            return {'performance_score': 0.0, 'improvement_rate': 0.0}
        
        lap_times_seconds = [lt.total_seconds() for lt in valid_laps]
        lap_numbers = list(range(len(lap_times_seconds)))
        
        # Calculate improvement rate
        slope, _, r_value, _, _ = stats.linregress(lap_numbers, lap_times_seconds) if len(lap_times_seconds) > 1 else (0, 0, 0, 0, 0)
        
        # Performance score (based on consistency and speed)
        avg_time = np.mean(lap_times_seconds)
        std_time = np.std(lap_times_seconds)
        cv = std_time / avg_time if avg_time > 0 else 0
        performance_score = max(0, 100 * (1 - cv))  # Higher is better
        
        return {
            'performance_score': float(performance_score),
            'improvement_rate': float(slope),  # Negative = improving
            'correlation_coefficient': float(r_value),
            'coefficient_of_variation': float(cv)
        }
    
    def _analyze_driver_sectors(self, driver_laps: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze sector performance for a driver"""
        sector_analysis = {}
        
        sectors = ['Sector1Time', 'Sector2Time', 'Sector3Time']
        sector_names = ['sector_1', 'sector_2', 'sector_3']
        
        for sector, name in zip(sectors, sector_names):
            sector_times = driver_laps[sector].dropna()
            
            if not sector_times.empty:
                sector_times_seconds = [st.total_seconds() for st in sector_times]
                
                sector_analysis[name] = {
                    'best_time': float(min(sector_times_seconds)),
                    'average_time': float(np.mean(sector_times_seconds)),
                    'worst_time': float(max(sector_times_seconds)),
                    'consistency': float(100 - (np.std(sector_times_seconds) / np.mean(sector_times_seconds) * 100)) if np.mean(sector_times_seconds) > 0 else 0,
                    'improvement_potential': float(max(sector_times_seconds) - min(sector_times_seconds))
                }
            else:
                sector_analysis[name] = {
                    'best_time': 0.0, 'average_time': 0.0, 'worst_time': 0.0,
                    'consistency': 0.0, 'improvement_potential': 0.0
                }
        
        return sector_analysis
    
    def _calculate_consistency_metrics(self, driver_laps: pd.DataFrame) -> Dict[str, float]:
        """Calculate consistency metrics for a driver"""
        valid_laps = driver_laps['LapTime'].dropna()
        
        if len(valid_laps) < 3:
            return {'consistency_score': 0.0, 'consistency_rank': 0.0}
        
        lap_times_seconds = [lt.total_seconds() for lt in valid_laps]
        
        # Various consistency measures
        cv = np.std(lap_times_seconds) / np.mean(lap_times_seconds) if np.mean(lap_times_seconds) > 0 else 0
        consistency_score = max(0, 100 * (1 - cv * 10))  # Scale factor
        
        # Quartile-based consistency
        q1, q3 = np.percentile(lap_times_seconds, [25, 75])
        iqr_consistency = 100 - ((q3 - q1) / np.median(lap_times_seconds) * 100) if np.median(lap_times_seconds) > 0 else 0
        
        return {
            'consistency_score': float(min(100.0, consistency_score)),
            'iqr_consistency': float(max(0.0, iqr_consistency)),
            'lap_time_std': float(np.std(lap_times_seconds)),
            'outlier_count': int(len([lt for lt in lap_times_seconds if abs(lt - np.mean(lap_times_seconds)) > 2 * np.std(lap_times_seconds)]))
        }
    
    def _extract_telemetry_metrics(self, telemetry: pd.DataFrame) -> Dict[str, float]:
        """Extract key telemetry metrics from fastest lap"""
        if telemetry.empty:
            return {}
        
        metrics = {
            'max_speed': float(telemetry['Speed'].max()),
            'avg_speed': float(telemetry['Speed'].mean()),
            'min_speed': float(telemetry['Speed'].min()),
            'max_throttle': float(telemetry['Throttle'].max()),
            'avg_throttle': float(telemetry['Throttle'].mean()),
            'max_brake': float(telemetry['Brake'].max()),
            'avg_brake': float(telemetry['Brake'].mean()),
            'max_rpm': float(telemetry['RPM'].max()),
            'avg_rpm': float(telemetry['RPM'].mean()),
            'max_gear': float(telemetry['nGear'].max()),
            'gear_changes': len(telemetry['nGear'].diff().dropna()[telemetry['nGear'].diff().dropna() != 0])
        }
        
        # Calculate time spent in different zones
        total_points = len(telemetry)
        metrics['full_throttle_percentage'] = float((telemetry['Throttle'] == 100).sum() / total_points * 100)
        metrics['braking_percentage'] = float((telemetry['Brake'] > 0).sum() / total_points * 100)
        metrics['coasting_percentage'] = float(((telemetry['Throttle'] == 0) & (telemetry['Brake'] == 0)).sum() / total_points * 100)
        
        # DRS usage if available
        if 'DRS' in telemetry.columns:
            metrics['drs_usage_percentage'] = float((telemetry['DRS'] > 0).sum() / total_points * 100)
        
        return metrics
    
    def _generate_comparative_insights(self, comparison_data: Dict, drivers: List[str]) -> Dict[str, Any]:
        """Generate comparative insights between drivers"""
        insights = {
            'fastest_driver': None,
            'most_consistent_driver': None,
            'best_sector_performers': {},
            'telemetry_leaders': {},
            'overall_rankings': {}
        }
        
        try:
            # Find fastest driver
            fastest_times = {}
            for driver in drivers:
                if driver in comparison_data['driver_statistics']:
                    fastest_times[driver] = comparison_data['driver_statistics'][driver].get('fastest_lap_time', float('inf'))
            
            if fastest_times:
                insights['fastest_driver'] = min(fastest_times, key=fastest_times.get)
            
            # Find most consistent driver
            consistency_scores = {}
            for driver in drivers:
                if driver in comparison_data['consistency_metrics']:
                    consistency_scores[driver] = comparison_data['consistency_metrics'][driver].get('consistency_score', 0)
            
            if consistency_scores:
                insights['most_consistent_driver'] = max(consistency_scores, key=consistency_scores.get)
            
            # Best sector performers
            for sector in ['sector_1', 'sector_2', 'sector_3']:
                sector_times = {}
                for driver in drivers:
                    if driver in comparison_data['sector_analysis']:
                        sector_data = comparison_data['sector_analysis'][driver].get(sector, {})
                        best_time = sector_data.get('best_time', float('inf'))
                        if best_time > 0:
                            sector_times[driver] = best_time
                
                if sector_times:
                    insights['best_sector_performers'][sector] = min(sector_times, key=sector_times.get)
            
            # Telemetry leaders
            telemetry_categories = ['max_speed', 'avg_speed', 'full_throttle_percentage']
            for category in telemetry_categories:
                category_values = {}
                for driver in drivers:
                    if driver in comparison_data['telemetry_comparison']:
                        value = comparison_data['telemetry_comparison'][driver].get(category, 0)
                        category_values[driver] = value
                
                if category_values:
                    insights['telemetry_leaders'][category] = max(category_values, key=category_values.get)
            
            # Overall rankings
            insights['overall_rankings'] = self._calculate_overall_rankings(comparison_data, drivers)
        
        except Exception as e:
            self.logger.warning(f"Error generating comparative insights: {str(e)}")
        
        return insights
    
    def _calculate_overall_rankings(self, comparison_data: Dict, drivers: List[str]) -> Dict[str, int]:
        """Calculate overall rankings for drivers"""
        rankings = {}
        
        try:
            # Create a composite score for each driver
            driver_scores = {}
            
            for driver in drivers:
                score = 0
                weight_total = 0
                
                # Speed ranking (40% weight)
                if driver in comparison_data['driver_statistics']:
                    fastest_time = comparison_data['driver_statistics'][driver].get('fastest_lap_time', 0)
                    if fastest_time > 0:
                        # Convert to score (lower time = higher score)
                        speed_score = max(0, 200 - fastest_time)
                        score += speed_score * 0.4
                        weight_total += 0.4
                
                # Consistency ranking (30% weight)
                if driver in comparison_data['consistency_metrics']:
                    consistency_score = comparison_data['consistency_metrics'][driver].get('consistency_score', 0)
                    score += consistency_score * 0.3
                    weight_total += 0.3
                
                # Performance metrics (30% weight)
                if driver in comparison_data['performance_metrics']:
                    performance_score = comparison_data['performance_metrics'][driver].get('performance_score', 0)
                    score += performance_score * 0.3
                    weight_total += 0.3
                
                # Normalize score
                if weight_total > 0:
                    driver_scores[driver] = score / weight_total
                else:
                    driver_scores[driver] = 0
            
            # Convert scores to rankings
            sorted_drivers = sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (driver, score) in enumerate(sorted_drivers):
                rankings[driver] = i + 1
        
        except Exception as e:
            self.logger.warning(f"Error calculating overall rankings: {str(e)}")
            # Fallback to simple alphabetical ranking
            for i, driver in enumerate(sorted(drivers)):
                rankings[driver] = i + 1
        
        return rankings
    
    def get_head_to_head_comparison(self, year: int, grand_prix: str, session: str, 
                                   driver1: str, driver2: str) -> Dict[str, Any]:
        """Get detailed head-to-head comparison between two drivers"""
        try:
            session_obj = fastf1.get_session(year, grand_prix, session)
            session_obj.load()
            
            driver1_laps = session_obj.laps.pick_drivers(driver1)
            driver2_laps = session_obj.laps.pick_drivers(driver2)
            
            if driver1_laps.empty or driver2_laps.empty:
                return {'error': 'Insufficient data for comparison'}
            
            comparison = {
                'lap_time_comparison': self._compare_lap_times(driver1_laps, driver2_laps, driver1, driver2),
                'sector_comparison': self._compare_sectors(driver1_laps, driver2_laps, driver1, driver2),
                'telemetry_comparison': self._compare_telemetry(driver1_laps, driver2_laps, driver1, driver2),
                'consistency_comparison': self._compare_consistency(driver1_laps, driver2_laps, driver1, driver2),
                'summary': self._generate_head_to_head_summary(driver1_laps, driver2_laps, driver1, driver2)
            }
            
            return {
                'head_to_head_comparison': comparison,
                'session_info': {
                    'year': year,
                    'grand_prix': grand_prix,
                    'session': session,
                    'driver1': driver1,
                    'driver2': driver2
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in head-to-head comparison: {str(e)}")
            return {'error': str(e)}
    
    def _compare_lap_times(self, laps1: pd.DataFrame, laps2: pd.DataFrame, 
                          driver1: str, driver2: str) -> Dict[str, Any]:
        """Compare lap times between two drivers"""
        try:
            valid_laps1 = laps1['LapTime'].dropna()
            valid_laps2 = laps2['LapTime'].dropna()
            
            if valid_laps1.empty or valid_laps2.empty:
                return {'error': 'No valid lap times for comparison'}
            
            times1 = [lt.total_seconds() for lt in valid_laps1]
            times2 = [lt.total_seconds() for lt in valid_laps2]
            
            comparison = {
                driver1: {
                    'fastest_lap': float(min(times1)),
                    'average_lap': float(np.mean(times1)),
                    'median_lap': float(np.median(times1))
                },
                driver2: {
                    'fastest_lap': float(min(times2)),
                    'average_lap': float(np.mean(times2)),
                    'median_lap': float(np.median(times2))
                },
                'differences': {
                    'fastest_lap_gap': float(min(times1) - min(times2)),
                    'average_lap_gap': float(np.mean(times1) - np.mean(times2)),
                    'median_lap_gap': float(np.median(times1) - np.median(times2))
                }
            }
            
            return comparison
            
        except Exception as e:
            return {'error': str(e)}
    
    def _compare_sectors(self, laps1: pd.DataFrame, laps2: pd.DataFrame, 
                        driver1: str, driver2: str) -> Dict[str, Any]:
        """Compare sector times between two drivers"""
        try:
            sectors = ['Sector1Time', 'Sector2Time', 'Sector3Time']
            sector_names = ['sector_1', 'sector_2', 'sector_3']
            
            comparison = {}
            
            for sector, name in zip(sectors, sector_names):
                sector1_times = laps1[sector].dropna()
                sector2_times = laps2[sector].dropna()
                
                if not sector1_times.empty and not sector2_times.empty:
                    times1 = [st.total_seconds() for st in sector1_times]
                    times2 = [st.total_seconds() for st in sector2_times]
                    
                    comparison[name] = {
                        driver1: {
                            'best_time': float(min(times1)),
                            'average_time': float(np.mean(times1))
                        },
                        driver2: {
                            'best_time': float(min(times2)),
                            'average_time': float(np.mean(times2))
                        },
                        'gap': {
                            'best_time_gap': float(min(times1) - min(times2)),
                            'average_time_gap': float(np.mean(times1) - np.mean(times2))
                        }
                    }
            
            return comparison
            
        except Exception as e:
            return {'error': str(e)}
    
    def _compare_telemetry(self, laps1: pd.DataFrame, laps2: pd.DataFrame, 
                          driver1: str, driver2: str) -> Dict[str, Any]:
        """Compare telemetry data between two drivers"""
        try:
            # Get fastest laps for telemetry comparison
            fastest1 = laps1.pick_fastest()
            fastest2 = laps2.pick_fastest()
            
            if fastest1 is None or fastest2 is None:
                return {'error': 'No fastest lap data available'}
            
            telemetry1 = fastest1.get_telemetry()
            telemetry2 = fastest2.get_telemetry()
            
            if telemetry1.empty or telemetry2.empty:
                return {'error': 'No telemetry data available'}
            
            comparison = {
                driver1: self._extract_telemetry_metrics(telemetry1),
                driver2: self._extract_telemetry_metrics(telemetry2),
                'differences': {}
            }
            
            # Calculate differences for key metrics
            for metric in ['max_speed', 'avg_speed', 'max_throttle', 'avg_throttle']:
                if metric in comparison[driver1] and metric in comparison[driver2]:
                    comparison['differences'][metric] = comparison[driver1][metric] - comparison[driver2][metric]
            
            return comparison
            
        except Exception as e:
            return {'error': str(e)}
    
    def _compare_consistency(self, laps1: pd.DataFrame, laps2: pd.DataFrame, 
                           driver1: str, driver2: str) -> Dict[str, Any]:
        """Compare consistency between two drivers"""
        try:
            consistency1 = self._calculate_consistency_metrics(laps1)
            consistency2 = self._calculate_consistency_metrics(laps2)
            
            comparison = {
                driver1: consistency1,
                driver2: consistency2,
                'more_consistent': driver1 if consistency1.get('consistency_score', 0) > consistency2.get('consistency_score', 0) else driver2
            }
            
            return comparison
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_head_to_head_summary(self, laps1: pd.DataFrame, laps2: pd.DataFrame, 
                                     driver1: str, driver2: str) -> Dict[str, Any]:
        """Generate summary of head-to-head comparison"""
        try:
            # Basic comparison
            valid_laps1 = laps1['LapTime'].dropna()
            valid_laps2 = laps2['LapTime'].dropna()
            
            if valid_laps1.empty or valid_laps2.empty:
                return {'error': 'Insufficient data for summary'}
            
            times1 = [lt.total_seconds() for lt in valid_laps1]
            times2 = [lt.total_seconds() for lt in valid_laps2]
            
            faster_driver = driver1 if min(times1) < min(times2) else driver2
            more_consistent = driver1 if np.std(times1) < np.std(times2) else driver2
            
            summary = {
                'faster_driver': faster_driver,
                'more_consistent_driver': more_consistent,
                'lap_time_advantage': abs(min(times1) - min(times2)),
                'consistency_advantage': abs(np.std(times1) - np.std(times2)),
                'overall_winner': self._determine_overall_winner(times1, times2, driver1, driver2)
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    def _determine_overall_winner(self, times1: List[float], times2: List[float], 
                                 driver1: str, driver2: str) -> str:
        """Determine overall winner based on speed and consistency"""
        try:
            # Weight speed (70%) and consistency (30%)
            speed_score1 = 100 / min(times1) if times1 else 0
            speed_score2 = 100 / min(times2) if times2 else 0
            
            consistency_score1 = 100 / (1 + np.std(times1)) if times1 else 0
            consistency_score2 = 100 / (1 + np.std(times2)) if times2 else 0
            
            overall_score1 = speed_score1 * 0.7 + consistency_score1 * 0.3
            overall_score2 = speed_score2 * 0.7 + consistency_score2 * 0.3
            
            return driver1 if overall_score1 > overall_score2 else driver2
            
        except Exception:
            return "Unable to determine"
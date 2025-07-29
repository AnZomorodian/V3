"""
Data Loader utility for F1 Analytics
Handles FastF1 data loading and caching
"""

import fastf1
import pandas as pd
import logging
from typing import Optional, Dict, Any
import os

class DataLoader:
    """FastF1 data loader with caching support"""
    
    def __init__(self, cache_dir: str = 'f1_cache'):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Enable FastF1 caching
        try:
            fastf1.Cache.enable_cache(cache_dir)
            self.logger.info(f"FastF1 cache enabled in {cache_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to enable cache: {e}")
    
    def load_session_data(self, year: int, gp: str, session: str = 'Race'):
        """Load F1 session data using FastF1 with timeout protection"""
        try:
            self.logger.info(f"Loading {year} {gp} {session} data...")
            
            # Load session with reduced data to prevent timeouts
            session_obj = fastf1.get_session(year, gp, session)
            
            # Load with minimal telemetry to reduce timeout risk
            try:
                session_obj.load(telemetry=False, weather=True, messages=True)
                self.logger.info("Loaded session data without telemetry to prevent timeout")
            except:
                # Fallback to basic load
                session_obj.load(telemetry=False, weather=False, messages=False)
                self.logger.warning("Loaded session data with minimal features")
            
            return session_obj
            
        except Exception as e:
            self.logger.error(f"Error loading session data: {str(e)}")
            # Try alternative session if available
            try:
                if session == 'Race':
                    self.logger.info("Trying Qualifying session as fallback...")
                    session_obj = fastf1.get_session(year, gp, 'Qualifying')
                    session_obj.load(telemetry=False, weather=False, messages=False)
                    return session_obj
            except:
                pass
            return None
    
    def get_driver_data(self, session_data, driver: str) -> Optional[pd.DataFrame]:
        """Get specific driver data from session"""
        try:
            if session_data is None:
                return None
            
            driver_laps = session_data.laps.pick_driver(driver)
            return driver_laps
            
        except Exception as e:
            self.logger.error(f"Error getting driver data: {str(e)}")
            return None
    
    def get_fastest_lap(self, session_data) -> Optional[pd.Series]:
        """Get the fastest lap from the session"""
        try:  
            if session_data is None:
                return None
            
            fastest_lap = session_data.laps.pick_fastest()
            return fastest_lap
            
        except Exception as e:
            self.logger.error(f"Error getting fastest lap: {str(e)}")
            return None
    
    def get_telemetry_data(self, session_data, driver: str, lap_number: int = None) -> Optional[pd.DataFrame]:
        """Get telemetry data for specific driver and lap"""
        try:
            if session_data is None:
                return None
            
            driver_laps = session_data.laps.pick_driver(driver)
            
            if lap_number is not None:
                lap = driver_laps[driver_laps['LapNumber'] == lap_number]
                if not lap.empty:
                    return lap.iloc[0].get_telemetry()
            else:
                # Get fastest lap telemetry
                fastest_lap = driver_laps.pick_fastest()
                if fastest_lap is not None:
                    return fastest_lap.get_telemetry()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting telemetry data: {str(e)}")
            return None
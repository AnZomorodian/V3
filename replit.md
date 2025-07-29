# F1 Analytics Dashboard

## Overview

This is a comprehensive Formula 1 analytics platform built with Flask that provides advanced performance analysis, real-time data streaming, and quantum-inspired analytics for F1 racing data. The application leverages FastF1 for data acquisition and offers multiple specialized analysis modules including quantum analytics, race strategy analysis, driver stress analysis, and real-time performance monitoring.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: Flask web application with SQLAlchemy ORM
- **Database**: SQLite by default (configurable via DATABASE_URL environment variable)
- **Data Source**: FastF1 library for official F1 timing and telemetry data
- **Analytics Engine**: Multiple specialized analysis modules with numpy, pandas, and scikit-learn

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap 5 dark theme
- **JavaScript**: Chart.js for data visualization
- **Styling**: Custom CSS with F1-themed design elements

### Application Structure
- **Entry Point**: `main.py` starts the Flask development server
- **Core Application**: `app.py` handles Flask app initialization, database setup, and configuration
- **Models**: Simple SQLAlchemy model for tracking analysis requests
- **Routes**: RESTful API endpoints for different analysis types
- **Analytics Modules**: Specialized analysis classes in the `analytics/` directory
- **Utilities**: Helper modules for data loading, JSON serialization, and constants

## Key Components

### Core Application (`app.py`)
- Flask application factory pattern
- SQLAlchemy database integration with declarative base
- Environment-based configuration for database and session management
- ProxyFix middleware for deployment compatibility

### Database Layer (`models.py`)
- `AnalysisRequest` model tracks user analysis requests
- Simple schema with year, grand prix, session, analysis type, and timestamp
- JSON serialization method for API responses

### API Layer (`routes.py`)
- RESTful endpoints for different analysis types
- Quantum analysis endpoint (`/api/quantum-analysis`)
- Request logging and error handling
- Integration with all analytics modules

### Analytics Modules
1. **Quantum Analytics** (`analytics/quantum_analytics.py`)
   - Quantum-inspired performance analysis algorithms
   - Advanced mathematical modeling with machine learning
   - Multiple analysis dimensions including lap optimization and strategy modeling

2. **Race Strategy Analyzer** (`analytics/race_strategy.py`)
   - Pit stop analysis and timing optimization
   - Tire strategy evaluation
   - Undercut/overcut opportunity analysis

3. **Real-time Analytics** (`analytics/real_time_analytics.py`)
   - Live session status monitoring
   - Real-time performance metrics
   - Gap analysis and live timing data

4. **Driver Stress Analysis** (`analytics/stress_index.py`)
   - Comprehensive stress index calculation
   - Sector-based stress analysis
   - Consistency and pressure moment identification

5. **Advanced Analytics** (`analytics/advanced_analytics.py`)
   - Performance metrics analysis
   - Tire degradation modeling
   - Comprehensive session analysis

### Data Layer (`utils/data_loader.py`)
- FastF1 integration for official F1 data
- Caching system for improved performance
- Session data loading and driver-specific data extraction
- Error handling and logging

### Utility Components
- **JSON Utils** (`utils/json_utils.py`): Handles conversion of complex pandas/numpy objects to JSON
- **Constants** (`utils/constants.py`): Team colors, driver mappings, and session types
- **Frontend Assets**: Bootstrap-based responsive design with F1 theming

## Data Flow

1. **User Request**: Frontend sends analysis request via REST API
2. **Request Logging**: Analysis request is logged to SQLite database
3. **Data Loading**: FastF1 library fetches official F1 timing and telemetry data
4. **Analysis Processing**: Specialized analytics modules process the data
5. **Result Serialization**: Complex data types converted to JSON-serializable format
6. **Response Delivery**: Results returned to frontend for visualization
7. **Data Visualization**: Chart.js renders interactive charts and graphs

## External Dependencies

### Core Dependencies
- **Flask**: Web framework and API server
- **SQLAlchemy**: Database ORM and query builder
- **FastF1**: Official F1 data API and timing data
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Scikit-learn**: Machine learning algorithms for predictive analytics

### Frontend Dependencies
- **Bootstrap 5**: UI framework with dark theme
- **Chart.js**: Interactive data visualization
- **Font Awesome**: Icon library for UI elements

### Analysis Dependencies
- **SciPy**: Statistical analysis and optimization
- **Random Forest & Neural Networks**: Advanced predictive modeling

## Deployment Strategy

### Development Environment
- Flask development server with debug mode enabled
- SQLite database for local development
- FastF1 caching enabled for performance optimization

### Production Considerations
- Environment variable configuration for database and session secrets
- ProxyFix middleware for reverse proxy compatibility
- Logging configuration for production monitoring
- Database connection pooling with health checks

### Environment Variables
- `DATABASE_URL`: Database connection string (defaults to SQLite)
- `SESSION_SECRET`: Flask session encryption key (defaults to development key)

### Performance Optimizations
- FastF1 data caching to reduce API calls
- Database connection pooling and health checks
- JSON serialization utilities for complex data types
- Efficient data processing with pandas vectorization

The application is designed to be easily deployable on platforms like Replit, Heroku, or similar cloud platforms with minimal configuration required.
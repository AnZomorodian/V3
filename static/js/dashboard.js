/**
 * F1 Analytics Dashboard JavaScript
 * Handles frontend interactions, API calls, and data visualization
 */

// Global variables
let currentAnalysisType = '';
let currentResults = null;
let performanceChart = null;
let lapTimeChart = null;

// Team colors for visualization
const TEAM_COLORS = {
    'Red Bull Racing': '#0600EF',
    'Mercedes': '#00D2BE',
    'Ferrari': '#DC143C',
    'McLaren': '#FF8700',
    'Alpine': '#0090FF',
    'Aston Martin': '#006F62',
    'Williams': '#005AFF',
    'AlphaTauri': '#2B4562',
    'Alfa Romeo': '#900000',
    'Haas': '#FFFFFF',
    'Kick Sauber': '#00FF00',
    'VCARB': '#6692FF'
};

// Tire compound colors
const TIRE_COLORS = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFFF33',
    'HARD': '#FFFFFF',
    'INTERMEDIATE': '#33FF33',
    'WET': '#3333FF'
};

/**
 * Utility Functions
 */

// Show loading indicator
function showLoading() {
    const loadingIndicator = document.getElementById('loadingIndicator');
    loadingIndicator.style.display = 'block';
    loadingIndicator.scrollIntoView({ behavior: 'smooth' });
}

// Hide loading indicator
function hideLoading() {
    document.getElementById('loadingIndicator').style.display = 'none';
}

// Get form parameters
function getFormParameters() {
    return {
        year: parseInt(document.getElementById('year').value),
        grand_prix: document.getElementById('grandPrix').value,
        session: document.getElementById('session').value,
        driver: document.getElementById('driver').value
    };
}

// Display error message
function displayError(message, container = 'resultsContent') {
    const content = document.getElementById(container);
    content.innerHTML = `
        <div class="error-message fade-in">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Error:</strong> ${message}
        </div>
    `;
}

// Display success message
function displaySuccess(message, container = 'resultsContent') {
    const content = document.getElementById(container);
    content.innerHTML = `
        <div class="success-message fade-in">
            <i class="fas fa-check-circle me-2"></i>
            ${message}
        </div>
    `;
}

// Format time duration
function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return 'N/A';

    const minutes = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(3);

    if (minutes > 0) {
        return `${minutes}:${secs.padStart(6, '0')}`;
    }
    return `${secs}s`;
}

// Format lap time from various formats
function formatLapTime(lapTime) {
    if (!lapTime || lapTime === 'N/A') return 'N/A';

    if (typeof lapTime === 'string') {
        // Already formatted
        if (lapTime.includes(':')) return lapTime;
        // Try to parse as float
        const parsed = parseFloat(lapTime);
        if (!isNaN(parsed)) {
            return formatTime(parsed);
        }
    }

    if (typeof lapTime === 'number') {
        return formatTime(lapTime);
    }

    return lapTime.toString();
}

// Get tire compound CSS class
function getTireCompoundClass(compound) {
    if (!compound) return 'tire-unknown';

    const compoundUpper = compound.toString().toUpperCase();
    if (compoundUpper.includes('SOFT')) return 'tire-soft';
    if (compoundUpper.includes('MEDIUM')) return 'tire-medium';
    if (compoundUpper.includes('HARD')) return 'tire-hard';
    if (compoundUpper.includes('INTERMEDIATE')) return 'tire-intermediate';
    if (compoundUpper.includes('WET')) return 'tire-wet';

    return 'tire-unknown';
}

// Get position badge class
function getPositionBadgeClass(position) {
    if (position === 1) return 'position-1';
    if (position === 2) return 'position-2';
    if (position === 3) return 'position-3';
    return 'bg-secondary';
}

/**
 * API Call Functions
 */

// Generic API call function
async function makeAPICall(endpoint, data) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        return result;
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

/**
 * Analysis Functions
 */

// Run Quantum Analysis
async function runQuantumAnalysis() {
    currentAnalysisType = 'quantum';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/quantum-analysis', params);

        currentResults = result;
        displayQuantumResults(result);
        showChartsSection();
        createQuantumCharts(result);

    } catch (error) {
        displayError(`Quantum analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Race Strategy Analysis
async function runRaceStrategy() {
    currentAnalysisType = 'strategy';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/race-strategy', params);

        currentResults = result;
        displayStrategyResults(result);
        showChartsSection();
        createStrategyCharts(result);

    } catch (error) {
        displayError(`Race strategy analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Real-time Analysis
async function runRealtimeAnalysis() {
    currentAnalysisType = 'realtime';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/realtime-status', params);

        currentResults = result;
        displayRealtimeResults(result);
        showChartsSection();
        createRealtimeCharts(result);

    } catch (error) {
        displayError(`Real-time analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Stress Analysis
async function runStressAnalysis() {
    currentAnalysisType = 'stress';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/stress-analysis', params);

        currentResults = result;
        displayStressResults(result);
        showChartsSection();
        createStressCharts(result);

    } catch (error) {
        displayError(`Stress analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Advanced Analysis
async function runAdvancedAnalysis() {
    currentAnalysisType = 'advanced';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/advanced-analysis', params);

        currentResults = result;
        displayAdvancedResults(result);
        showChartsSection();
        createAdvancedCharts(result);

    } catch (error) {
        displayError(`Advanced analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Weather Analysis
async function runWeatherAnalysis() {
    currentAnalysisType = 'weather';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/weather-analysis', params);

        currentResults = result;
        displayWeatherResults(result);
        showWeatherSection();

    } catch (error) {
        displayError(`Weather analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Driver Comparison
async function runDriverComparison() {
    currentAnalysisType = 'comparison';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/driver-comparison', params);

        currentResults = result;
        displayComparisonResults(result);
        showComparisonSection();

    } catch (error) {
        console.error('Driver comparison error:', error);
        displayError(`Driver comparison failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

/**
 * Results Display Functions
 */

// Display Quantum Analysis Results
function displayQuantumResults(data) {
    const container = document.getElementById('resultsContent');

    let html = `
        <div class="fade-in">
            <h4><i class="fas fa-atom me-2 text-primary"></i>Quantum Performance Analysis Results</h4>
            <div class="row">
    `;

    // Quantum Lap Optimization
    if (data.quantum_lap_optimization && data.quantum_lap_optimization.quantum_states) {
        html += `
            <div class="col-md-6">
                <div class="card quantum-card mb-3">
                    <div class="card-header">
                        <h5><i class="fas fa-cog me-2"></i>Quantum Lap Optimization</h5>
                    </div>
                    <div class="card-body">
        `;

        Object.entries(data.quantum_lap_optimization.quantum_states).forEach(([driver, state]) => {
            html += `
                <div class="driver-row">
                    <strong>${driver}</strong>
                    <div class="row mt-2">
                        <div class="col-sm-6">
                            <div class="metric-card">
                                <div class="metric-value">${formatTime(state.base_state_laptime)}</div>
                                <div class="metric-label">Base State Lap Time</div>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="metric-card">
                                <div class="metric-value">${(state.quantum_efficiency_score * 100).toFixed(1)}%</div>
                                <div class="metric-label">Quantum Efficiency</div>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-sm-6">
                            <div class="metric-card">
                                <div class="metric-value">${(state.optimal_state_probability * 100).toFixed(1)}%</div>
                                <div class="metric-label">Optimal State Probability</div>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="metric-card">
                                <div class="metric-value">${state.coherence_factor.toFixed(3)}</div>
                                <div class="metric-label">Coherence Factor</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        html += `
                    </div>
                </div>
            </div>
        `;
    }

    // Multiverse Strategy Analysis
    if (data.multiverse_strategy_modeling && data.multiverse_strategy_modeling.quantum_strategy_selection) {
        html += `
            <div class="col-md-6">
                <div class="card quantum-card mb-3">
                    <div class="card-header">
                        <h5><i class="fas fa-universe me-2"></i>Multiverse Strategy Modeling</h5>
                    </div>
                    <div class="card-body">
        `;

        Object.entries(data.multiverse_strategy_modeling.quantum_strategy_selection).forEach(([driver, strategy]) => {
            html += `
                <div class="driver-row">
                    <strong>${driver}</strong>
                    <div class="mt-2">
                        <span class="badge bg-primary">${strategy.optimal_strategy}</span>
                        <div class="mt-1">
                            <small>Strategy Score: ${strategy.strategy_score.toFixed(2)}</small>
                            <br>
                            <small>Dimensional Advantage: ${strategy.dimensional_advantage.toFixed(2)}</small>
                        </div>
                    </div>
                </div>
            `;
        });

        html += `
                    </div>
                </div>
            </div>
        `;
    }

    html += `
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// Display Strategy Results
function displayStrategyResults(data) {
    const container = document.getElementById('resultsContent');

    let html = `
        <div class="fade-in">
            <h4><i class="fas fa-chess me-2 text-success"></i>Race Strategy Analysis Results</h4>
            <div class="row">
    `;

    // Pit Stop Analysis
    if (data.pit_stop_analysis) {
        html += `
            <div class="col-12">
                <div class="card strategy-card mb-3">
                    <div class="card-header">
                        <h5><i class="fas fa-tools me-2"></i>Pit Stop Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table data-table">
                                <thead>
                                    <tr>
                                        <th>Driver</th>
                                        <th>Total Stops</th>
                                        <th>Strategy Type</th>
                                        <th>Avg Duration</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
        `;

        Object.entries(data.pit_stop_analysis).forEach(([driver, analysis]) => {
            html += `
                <tr>
                    <td><strong>${driver}</strong></td>
                    <td>${analysis.total_pit_stops}</td>
                    <td><span class="badge bg-info">${analysis.strategy_type}</span></td>
                    <td>${analysis.avg_pit_duration ? formatTime(analysis.avg_pit_duration) : 'N/A'}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-light" onclick="showPitStopDetails('${driver}')">
                            <i class="fas fa-eye me-1"></i>View
                        </button>
                    </td>
                </tr>
            `;
        });

        html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Strategy Effectiveness
    if (data.strategy_effectiveness) {
        html += `
            <div class="col-md-6">
                <div class="card strategy-card mb-3">
                    <div class="card-header">
                        <h5><i class="fas fa-trophy me-2"></i>Strategy Effectiveness</h5>
                    </div>
                    <div class="card-body">
        `;

        Object.entries(data.strategy_effectiveness).forEach(([driver, effectiveness]) => {
            const positionChange = effectiveness.position_change;
            const changeClass = positionChange > 0 ? 'text-success' : positionChange < 0 ? 'text-danger' : 'text-muted';
            const changeIcon = positionChange > 0 ? 'fa-arrow-up' : positionChange < 0 ? 'fa-arrow-down' : 'fa-minus';

            html += `
                <div class="driver-row">
                    <div class="d-flex justify-content-between align-items-center">
                        <strong>${driver}</strong>
                        <span class="${changeClass}">
                            <i class="fas ${changeIcon} me-1"></i>
                            ${positionChange > 0 ? '+' : ''}${positionChange}
                        </span>
                    </div>
                    <div class="row mt-2">
                        <div class="col-4">
                            <small>Grid: P${effectiveness.grid_position}</small>
                        </div>
                        <div class="col-4">
                            <small>Final: P${effectiveness.final_position}</small>
                        </div>
                        <div class="col-4">
                            <small>Points: ${effectiveness.points_scored}</small>
                        </div>
                    </div>
                    <div class="mt-2">
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar" style="width: ${effectiveness.effectiveness_rating}%"></div>
                        </div>
                        <small>Effectiveness: ${effectiveness.effectiveness_rating}%</small>
                    </div>
                </div>
            `;
        });

        html += `
                    </div>
                </div>
            </div>
        `;
    }

    html += `
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// Display Real-time Results
function displayRealtimeResults(data) {
    const container = document.getElementById('resultsContent');

    let html = `
        <div class="fade-in">
            <h4><i class="fas fa-broadcast-tower me-2 text-info"></i>Real-time Session Status</h4>
    `;

    if (data.status === 'live' && data.live_standings) {
        html += `
            <div class="row">
                <div class="col-12">
                    <div class="card mb-3">
                        <div class="card-header">
                            <h5><i class="fas fa-flag-checkered me-2"></i>Live Standings</h5>
                            <small class="text-muted">
                                ${data.session_info ? `${data.session_info.year} ${data.session_info.grand_prix}` : ''}
                            </small>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table data-table">
                                    <thead>
                                        <tr>
                                            <th>Pos</th>
                                            <th>Driver</th>
                                            <th>Team</th>
                                            <th>Last Lap</th>
                                            <th>Compound</th>
                                            <th>Tire Life</th>
                                        </tr>
                                    </thead>
                                    <tbody>
        `;

        data.live_standings.forEach(standing => {
            html += `
                <tr>
                    <td>
                        <span class="position-badge ${getPositionBadgeClass(standing.position)}">
                            ${standing.position}
                        </span>
                    </td>
                    <td><strong>${standing.driver}</strong></td>
                    <td>
                        <span style="color: ${standing.team_color || '#808080'}">
                            ${standing.team}
                        </span>
                    </td>
                    <td>${formatLapTime(standing.last_lap_time)}</td>
                    <td>
                        <span class="tire-compound ${getTireCompoundClass(standing.compound)}">
                            ${standing.compound || 'N/A'}
                        </span>
                    </td>
                    <td>${standing.tire_life || 0}</td>
                </tr>
            `;
        });

        html += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } else {
        html += `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${data.message || 'No live session data available'}
            </div>
        `;
    }

    html += `</div>`;
    container.innerHTML = html;
}

// Display Stress Analysis Results
function displayStressResults(data) {
    const container = document.getElementById('resultsContent');
    const driver = getFormParameters().driver;

    let html = `
        <div class="fade-in">
            <h4><i class="fas fa-heartbeat me-2 text-warning"></i>Driver Stress Analysis: ${driver}</h4>
    `;

    if (data.error) {
        html += `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                ${data.error}
            </div>
        `;
    } else {
        // Overall Stress Index
        if (data.overall_stress_index) {
            const stressIndex = data.overall_stress_index.index || 0;
            const stressRating = data.overall_stress_index.rating || 'unknown';

            html += `
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="card stress-card">
                            <div class="card-body text-center">
                                <div class="metric-value" style="color: ${getStressColor(stressIndex)}">${stressIndex.toFixed(1)}</div>
                                <div class="metric-label">Overall Stress Index</div>
                                <div class="stress-indicator mt-2">
                                    <div class="stress-marker" style="left: ${stressIndex}%"></div>
                                </div>
                                <small class="text-muted mt-1 d-block">Rating: ${stressRating}</small>
                            </div>
                        </div>
                    </div>
            `;

            // Sector Stress Analysis
            if (data.sector_stress_analysis) {
                html += `
                    <div class="col-md-8">
                        <div class="card stress-card">
                            <div class="card-header">
                                <h5><i class="fas fa-map me-2"></i>Sector Stress Analysis</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                `;

                Object.entries(data.sector_stress_analysis).forEach(([sector, analysis]) => {
                    if (sector !== 'error') {
                        html += `
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-value" style="color: ${getStressColor(analysis.stress_index)}">${analysis.stress_index.toFixed(1)}</div>
                                    <div class="metric-label">${sector.toUpperCase()}</div>
                                    <small>CV: ${(analysis.coefficient_of_variation * 100).toFixed(2)}%</small>
                                </div>
                            </div>
                        `;
                    }
                });

                html += `
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                `;
            }
        }

        // Consistency Analysis
       if (data.consistency_stress && !data.consistency_stress.error) {
            html += `
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card stress-card">
                            <div class="card-header">
                                <h5><i class="fas fa-chart-line me-2"></i>Consistency Analysis</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-3">
                                        <div class="metric-card">
                                            <div class="metric-value">${(data.consistency_stress.overall_consistency * 100).toFixed(2)}%</div>
                                            <div class="metric-label">Overall Consistency</div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="metric-card">
                                            <div class="metric-value">${data.consistency_stress.consistency_rating}</div>
                                            <div class="metric-label">Consistency Rating</div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="metric-card">
                                            <div class="metric-value">${data.consistency_stress.breakdown_incidents ? data.consistency_stress.breakdown_incidents.length : 0}</div>
                                            <div class="metric-label">Breakdown Incidents</div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="text-center">
                                            <button class="btn btn-sm btn-outline-light" onclick="showConsistencyDetails()">
                                                <i class="fas fa-info-circle me-1"></i>Details
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    html += `</div>`;
    container.innerHTML = html;
}

// Display Advanced Analysis Results
function displayAdvancedResults(data) {
    const container = document.getElementById('resultsContent');

    let html = `
        <div class="fade-in">
            <h4><i class="fas fa-cogs me-2 text-secondary"></i>Advanced Session Analysis</h4>
    `;

    if (!data) {
        html += `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                No advanced analysis data available for the selected session.
            </div>
        `;
    } else {
        // Performance Analysis
        if (data.performance_analysis) {
            html += `
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card mb-3">
                            <div class="card-header">
                                <h5><i class="fas fa-tachometer-alt me-2"></i>Performance Metrics</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table data-table">
                                        <thead>
                                            <tr>
                                                <th>Driver</th>
                                                <th>Fastest Lap</th>
                                                <th>Average Lap</th>
                                                <th>Total Laps</th>
                                                <th>Max Speed</th>
                                                <th>Consistency</th>
                                                <th>Overtakes</th>
                                                <th>Defenses</th>
                                            </tr>
                                        </thead>
                                        <tbody>
            `;

            Object.entries(data.performance_analysis).forEach(([driver, metrics]) => {
                html += `
                    <tr>
                        <td><strong>${driver}</strong></td>
                        <td>${formatLapTime(metrics.fastest_lap_time)}</td>
                        <td>${formatLapTime(metrics.average_lap_time)}</td>
                        <td>${metrics.total_laps}</td>
                        <td>${metrics.max_speed ? metrics.max_speed.toFixed(1) + ' km/h' : 'N/A'}</td>
                        <td>${metrics.consistency_score ? metrics.consistency_score.toFixed(2) : 'N/A'}</td>
                        <td>${metrics.overtakes || 'N/A'}</td>
                        <td>${metrics.defenses || 'N/A'}</td>
                    </tr>
                `;
            });

            html += `
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Consistency Analysis
        if (data.consistency_analysis) {
            html += `
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card mb-3">
                            <div class="card-header">
                                <h5><i class="fas fa-chart-bar me-2"></i>Lap Time Consistency Analysis</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
            `;

            Object.entries(data.consistency_analysis).forEach(([driver, consistency]) => {
                html += `
                    <div class="col-md-4 mb-3">
                        <div class="metric-card">
                            <strong>${driver}</strong>
                            <div class="mt-2">
                                <div class="metric-value text-primary">${consistency.consistency_score.toFixed(1)}</div>
                                <div class="metric-label">Consistency Score</div>
                                <small>CV: ${(consistency.coefficient_of_variation * 100).toFixed(2)}%</small>
                            </div>
                            <div>
                                <button class="btn btn-sm btn-outline-light mt-2" onclick="showLapTimeDetails('${driver}')">
                                    <i class="fas fa-info-circle me-1"></i>Lap Time Details
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += `
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    html += `</div>`;
    container.innerHTML = html;
}

// Display Weather Results
function displayWeatherResults(data) {
    const container = document.getElementById('weatherContent');

    let html = `
        <div class="fade-in">
            <h4><i class="fas fa-cloud-sun me-2 text-info"></i>Weather Impact Analysis</h4>
            <div class="row">
    `;

    if (data.weather_conditions) {
        html += `
            <div class="col-md-6">
                <div class="card weather-card mb-3">
                    <div class="card-header">
                        <h5><i class="fas fa-thermometer-half me-2"></i>Track Conditions</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <div class="metric-card">
                                    <div class="metric-value">${data.weather_conditions.avg_air_temp}°C</div>
                                    <div class="metric-label">Air Temperature</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="metric-card">
                                    <div class="metric-value">${data.weather_conditions.avg_track_temp}°C</div>
                                    <div class="metric-label">Track Temperature</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    html += `
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// Display Comparison Results
function displayComparisonResults(data) {
    const container = document.getElementById('comparisonContent');

    let html = `
        <div class="fade-in">
            <h4><i class="fas fa-users me-2 text-warning"></i>Driver Head-to-Head Comparison</h4>
            <div class="row">
    `;

    if (data.comparison_data) {
        html += `
            <div class="col-12">
                <div class="card comparison-card mb-3">
                    <div class="card-header">
                        <h5><i class="fas fa-balance-scale me-2"></i>Performance Comparison</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table data-table">
                                <thead>
                                    <tr>
                                        <th>Driver</th>
                                        <th>Best Lap</th>
                                        <th>Avg Lap</th>
                                        <th>Gap to Fastest</th>
                                        <th>Top Speed</th>
                                        <th>Position</th>
                                    </tr>
                                </thead>
                                <tbody>
        `;

        data.comparison_data.forEach(driver => {
            html += `
                <tr>
                    <td><strong>${driver.driver}</strong></td>
                    <td>${formatLapTime(driver.best_lap)}</td>
                    <td>${formatLapTime(driver.avg_lap)}</td>
                    <td>${driver.gap_to_fastest || 'N/A'}</td>
                    <td>${driver.top_speed !== 'N/A' && driver.top_speed ? driver.top_speed + ' km/h' : 'N/A'}</td>
                    <td>${driver.position}</td>
                </tr>
            `;
        });

        html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    html += `
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// Show sections
function showWeatherSection() {
    document.getElementById('weatherSection').style.display = 'block';
    document.getElementById('weatherSection').scrollIntoView({ behavior: 'smooth' });
}

function showComparisonSection() {
    document.getElementById('comparisonSection').style.display = 'block';
    document.getElementById('comparisonSection').scrollIntoView({ behavior: 'smooth' });
}

// Utility Functions
function showLoading() {
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function hideLoading() {
    document.getElementById('loadingIndicator').style.display = 'none';
}

/**
 * Chart Functions
 */

// Show charts section
function showChartsSection() {
    document.getElementById('chartsSection').style.display = 'block';
}

// Destroy existing charts
function destroyExistingCharts() {
    if (performanceChart) {
        performanceChart.destroy();
        performanceChart = null;
    }
    if (lapTimeChart) {
        lapTimeChart.destroy();
        lapTimeChart = null;
    }
}

// Create Quantum Charts
function createQuantumCharts(data) {
    destroyExistingCharts();

    // Performance Chart
    if (data.quantum_lap_optimization && data.quantum_lap_optimization.quantum_states) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const drivers = Object.keys(data.quantum_lap_optimization.quantum_states);
        const efficiencyScores = drivers.map(driver => 
            data.quantum_lap_optimization.quantum_states[driver].quantum_efficiency_score * 100
        );

        performanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: drivers,
                datasets: [{
                    label: 'Quantum Efficiency Score (%)',
                    data: efficiencyScores,
                    backgroundColor: 'rgba(6, 0, 239, 0.7)',
                    borderColor: 'rgba(6, 0, 239, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Quantum Efficiency Scores',
                        color: 'white'
                    },
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }
}

// Create Strategy Charts
function createStrategyCharts(data) {
    destroyExistingCharts();

    // Strategy Effectiveness Chart
    if (data.strategy_effectiveness) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const drivers = Object.keys(data.strategy_effectiveness);
        const effectiveness = drivers.map(driver => 
            data.strategy_effectiveness[driver].effectiveness_rating
        );

        performanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: drivers,
                datasets: [{
                    label: 'Strategy Effectiveness (%)',
                    data: effectiveness,
                    backgroundColor: 'rgba(40, 167, 69, 0.7)',
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Strategy Effectiveness',
                        color: 'white'
                    },
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }
}

// Create Real-time Charts
function createRealtimeCharts(data) {
    destroyExistingCharts();

    if (data.live_standings) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const drivers = data.live_standings.map(s => s.driver);
        const positions = data.live_standings.map(s => s.position);

        performanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: drivers,
                datasets: [{
                    label: 'Current Position',
                    data: positions,
                    backgroundColor: drivers.map(driver => 
                        TEAM_COLORS[data.live_standings.find(s => s.driver === driver)?.team] || '#808080'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Current Standings',
                        color: 'white'
                    },
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    y: {
                        reverse: true,
                        beginAtZero: true,
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }
}

// Create Stress Charts
function createStressCharts(data) {
    destroyExistingCharts();

    // Sector Stress Chart
    if (data.sector_stress_analysis) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const sectors = Object.keys(data.sector_stress_analysis).filter(key => key !== 'error');
        const stressValues = sectors.map(sector => 
            data.sector_stress_analysis[sector].stress_index
        );

        performanceChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: sectors.map(s => s.toUpperCase()),
                datasets: [{
                    label: 'Stress Index',
                    data: stressValues,
                    backgroundColor: 'rgba(255, 193, 7, 0.2)',
                    borderColor: 'rgba(255, 193, 7, 1)',
                    pointBackgroundColor: 'rgba(255, 193, 7, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(255, 193, 7, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Sector Stress Analysis',
                        color: 'white'
                    },
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: 'white'
                        },
                        ticks: {
                            color: 'white'
                        }
                    }
                }
            }
        });
    }
}

// Create Advanced Charts
function createAdvancedCharts(data) {
    destroyExistingCharts();

    // Performance Metrics Chart
    if (data.performance_analysis) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const drivers = Object.keys(data.performance_analysis);
        const lapCounts = drivers.map(driver => 
            data.performance_analysis[driver].total_laps
        );

        performanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: drivers,
                datasets: [{
                    label: 'Total Laps',
                    data: lapCounts,
                    backgroundColor: 'rgba(108, 117, 125, 0.7)',
                    borderColor: 'rgba(108, 117, 125, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Session Lap Count',
                        color: 'white'
                    },
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }
}

/**
 * Utility Functions for Analysis
 */

// Get stress level color
function getStressColor(stressLevel) {
    if (stressLevel < 20) return '#28a745'; // Green
    if (stressLevel < 40) return '#ffc107'; // Yellow
    if (stressLevel < 60) return '#fd7e14'; // Orange
    if (stressLevel < 80) return '#dc3545'; // Red
    return '#6f42c1'; // Purple
}

// Show pit stop details (placeholder)
function showPitStopDetails(driver) {
    alert(`Detailed pit stop analysis for ${driver} would be shown here.`);
}

// Show consistency details (placeholder)
function showConsistencyDetails() {
    alert('Detailed consistency breakdown would be shown here.');
}

//showLapTimeDetails
function showLapTimeDetails(driver) {
    alert(`Detailed lap time details for ${driver} would be shown here.`);
}
/**
 * Analysis History Functions
 */

/**
 * Enhanced Analytics Functions
 */

// Run Track Analysis
async function runTrackAnalysis() {
    currentAnalysisType = 'track-analysis';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/track-analysis', params);

        currentResults = result;
        displayEnhancedResults(result, 'Track Analysis Results');
        showChartsSection();

    } catch (error) {
        displayError(`Track analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Tire Performance Analysis
async function runTirePerformance() {
    currentAnalysisType = 'tire-performance';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/tire-performance', params);

        currentResults = result;
        displayEnhancedResults(result, 'Tire Performance Analysis');
        showChartsSection();

    } catch (error) {
        displayError(`Tire performance analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Downforce Analysis
async function runDownforceAnalysis() {
    currentAnalysisType = 'downforce-analysis';
    showLoading();

    try {
        const params = getFormParameters();
        const result = await makeAPICall('/api/downforce-analysis', params);

        currentResults = result;
        displayEnhancedResults(result, 'Downforce Analysis Results');
        showChartsSection();

    } catch (error) {
        displayError(`Downforce analysis failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Run Comprehensive Driver Comparison
async function runComprehensiveComparison() {
    currentAnalysisType = 'comprehensive-comparison';
    showLoading();

    try {
        const params = getFormParameters();
        // Add multiple top drivers for comprehensive comparison
        params.drivers = ['VER', 'HAM', 'LEC', 'NOR'];
        const result = await makeAPICall('/api/comprehensive-driver-comparison', params);

        currentResults = result;
        displayEnhancedResults(result, 'Multi-Driver Comparison');
        showChartsSection();

    } catch (error) {
        displayError(`Comprehensive driver comparison failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Enhanced Results Display Function
function displayEnhancedResults(data, title) {
    const resultsDiv = document.getElementById('resultsContent');
    
    let html = `
        <div class="analysis-results fade-in">
            <h4 class="mb-4 text-primary">
                <i class="fas fa-chart-line me-2"></i>${title}
            </h4>
    `;

    if (data.error) {
        html += `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Error:</strong> ${data.error}
            </div>
        `;
    } else {
        // Display different types of analysis results
        if (data.track_analysis) {
            html += displayTrackAnalysisResults(data.track_analysis);
        } else if (data.compound_performance) {
            html += displayTirePerformanceResults(data);
        } else if (data.aerodynamic_efficiency) {
            html += displayDownforceResults(data);
        } else if (data.driver_comparison) {
            html += displayDriverComparisonResults(data.driver_comparison);
        } else {
            // Default display for other analysis types
            html += '<div class="alert alert-success">';
            html += '<i class="fas fa-check-circle me-2"></i>';
            html += 'Analysis completed successfully. View detailed results below:';
            html += '</div>';
            html += '<pre class="bg-dark text-light p-3 rounded">';
            html += JSON.stringify(data, null, 2);
            html += '</pre>';
        }
    }

    html += '</div>';
    resultsDiv.innerHTML = html;
    
    // Show charts section if we have chart data
    if (data.charts || data.visualizations) {
        document.getElementById('chartsSection').style.display = 'block';
    }
}

function displayTrackAnalysisResults(trackData) {
    let html = '<div class="row fade-in-up">';
    
    if (trackData.track_info) {
        const distance = trackData.track_info.total_distance || 0;
        const distanceKm = (distance / 1000).toFixed(2);
        const turnCount = trackData.track_info.turn_count || 'Unknown';
        const drsZones = trackData.track_info.drs_zones || 0;
        
        html += `
            <div class="col-lg-4 col-md-6 mb-4">
                <div class="data-card track-info-card glow-effect">
                    <div class="result-card-header">
                        <i class="fas fa-road me-2"></i>Track Information
                    </div>
                    <div class="mt-3">
                        <div class="metric-display interactive-stat" data-tooltip="Total circuit length">
                            <div class="metric-value">${distanceKm} km</div>
                            <div class="metric-label">Track Distance</div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-6">
                                <div class="metric-display interactive-stat" data-tooltip="Number of corners/turns">
                                    <div class="metric-value">${turnCount}</div>
                                    <div class="metric-label">Turns</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="metric-display interactive-stat" data-tooltip="DRS activation zones">
                                    <div class="metric-value">${drsZones}</div>
                                    <div class="metric-label">DRS Zones</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    if (trackData.speed_analysis) {
        const maxSpeed = trackData.speed_analysis.max_speed || 0;
        const avgSpeed = trackData.speed_analysis.avg_speed || 0;
        const highSpeedPct = trackData.speed_analysis.high_speed_percentage || 0;
        const speedEfficiency = ((avgSpeed / maxSpeed) * 100).toFixed(1);
        
        html += `
            <div class="col-lg-4 col-md-6 mb-4">
                <div class="data-card speed-analysis-card">
                    <div class="result-card-header">
                        <i class="fas fa-tachometer-alt me-2"></i>Speed Analysis
                    </div>
                    <div class="mt-3">
                        <div class="metric-display interactive-stat" data-tooltip="Highest speed recorded on track">
                            <div class="metric-value">${maxSpeed.toFixed(1)}</div>
                            <div class="metric-label">Max Speed (km/h)</div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-6">
                                <div class="metric-display interactive-stat" data-tooltip="Average speed throughout lap">
                                    <div class="metric-value">${avgSpeed.toFixed(1)}</div>
                                    <div class="metric-label">Avg Speed</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="metric-display interactive-stat" data-tooltip="Percentage of lap at high speed">
                                    <div class="metric-value">${highSpeedPct.toFixed(1)}%</div>
                                    <div class="metric-label">High Speed</div>
                                </div>
                            </div>
                        </div>
                        <div class="mt-3">
                            <div class="progress-modern">
                                <div class="progress-bar-modern" style="width: ${speedEfficiency}%"></div>
                            </div>
                            <div class="text-center mt-2">
                                <small>Speed Efficiency: ${speedEfficiency}%</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Add track difficulty indicator
    html += `
        <div class="col-lg-4 col-md-12 mb-4">
            <div class="data-card sector-card">
                <div class="result-card-header">
                    <i class="fas fa-chart-line me-2"></i>Track Characteristics
                </div>
                <div class="mt-3">
                    <div class="metric-display interactive-stat" data-tooltip="Track difficulty rating">
                        <div class="metric-value">${calculateTrackDifficulty(trackData)}</div>
                        <div class="metric-label">Difficulty Rating</div>
                    </div>
                    <div class="mt-3">
                        <div class="status-indicator ${getTrackTypeClass(trackData)}">
                            ${getTrackType(trackData)}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    html += '</div>';
    return html;
}

// Helper functions for enhanced track analysis
function calculateTrackDifficulty(trackData) {
    if (!trackData.speed_analysis) return 'N/A';
    
    const speedVariance = trackData.speed_analysis.speed_variance || 0;
    const highSpeedPct = trackData.speed_analysis.high_speed_percentage || 0;
    
    const difficulty = Math.min(10, Math.max(1, (speedVariance / 1000 + (100 - highSpeedPct) / 20)));
    return difficulty.toFixed(1) + '/10';
}

function getTrackType(trackData) {
    if (!trackData.speed_analysis) return 'Unknown';
    
    const highSpeedPct = trackData.speed_analysis.high_speed_percentage || 0;
    
    if (highSpeedPct > 70) return 'High-Speed Circuit';
    if (highSpeedPct > 40) return 'Balanced Circuit';
    return 'Technical Circuit';
}

function getTrackTypeClass(trackData) {
    if (!trackData.speed_analysis) return 'status-poor';
    
    const highSpeedPct = trackData.speed_analysis.high_speed_percentage || 0;
    
    if (highSpeedPct > 70) return 'status-excellent';
    if (highSpeedPct > 40) return 'status-good';
    return 'status-poor';
}

function displayTirePerformanceResults(tireData) {
    let html = '<div class="row fade-in-up">';
    
    if (tireData.compound_performance && tireData.compound_performance.compound_data) {
        html += `
            <div class="col-12 mb-4">
                <div class="analytics-card">
                    <div class="result-card-header">
                        <i class="fas fa-circle me-2"></i>Tire Compound Performance Analysis
                    </div>
                </div>
            </div>
        `;
        
        Object.entries(tireData.compound_performance.compound_data).forEach(([compound, data]) => {
            const compoundClass = getTireCompoundClass(compound);
            const avgTime = data.average_lap_time || 0;
            const bestTime = data.best_lap_time || 0;
            const degradation = avgTime > 0 && bestTime > 0 ? ((avgTime - bestTime) / bestTime * 100).toFixed(2) : 0;
            const performance = avgTime > 0 ? Math.max(0, 100 - (avgTime - 60)).toFixed(1) : 0;
            
            html += `
                <div class="col-lg-4 col-md-6 mb-4 slide-in-right">
                    <div class="data-card ${compoundClass}">
                        <div class="result-card-header">
                            <i class="fas fa-circle me-2"></i>${compound} Compound
                        </div>
                        <div class="mt-3">
                            <div class="row">
                                <div class="col-6">
                                    <div class="metric-display interactive-stat" data-tooltip="Total laps completed on this compound">
                                        <div class="metric-value">${data.total_laps}</div>
                                        <div class="metric-label">Total Laps</div>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="metric-display interactive-stat" data-tooltip="Number of drivers who used this compound">
                                        <div class="metric-value">${data.drivers_used}</div>
                                        <div class="metric-label">Drivers</div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mt-3">
                                <div class="metric-display interactive-stat" data-tooltip="Fastest lap time on this compound">
                                    <div class="metric-value">${formatLapTime(data.best_lap_time)}</div>
                                    <div class="metric-label">Best Lap Time</div>
                                </div>
                            </div>
                            
                            <div class="mt-3">
                                <div class="metric-display interactive-stat" data-tooltip="Average lap time across all drivers">
                                    <div class="metric-value">${formatLapTime(data.average_lap_time)}</div>
                                    <div class="metric-label">Average Time</div>
                                </div>
                            </div>
                            
                            <div class="mt-3">
                                <div class="progress-modern">
                                    <div class="progress-bar-modern" style="width: ${Math.min(100, performance)}%"></div>
                                </div>
                                <div class="text-center mt-2">
                                    <small>Performance Rating: ${performance}%</small>
                                </div>
                            </div>
                            
                            <div class="mt-3 text-center">
                                <div class="badge ${degradation > 2 ? 'status-poor' : degradation > 1 ? 'status-good' : 'status-excellent'}">
                                    Degradation: ${degradation}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
    }

    html += '</div>';
    return html;
}



function displayDriverComparisonResults(comparisonData) {
    let html = '<div class="row fade-in-up">';
    
    if (comparisonData.comparative_analysis) {
        const analysis = comparisonData.comparative_analysis;
        html += `
            <div class="col-12 mb-4">
                <div class="analytics-card">
                    <div class="result-card-header">
                        <i class="fas fa-trophy me-2"></i>Head-to-Head Comparison Analysis
                    </div>
                    <div class="mt-3 p-3">
                        <div class="row text-center">
                            <div class="col-md-6">
                                <div class="metric-display status-excellent interactive-stat" data-tooltip="Driver with the fastest lap time">
                                    <div class="metric-value"><i class="fas fa-crown me-2"></i>${analysis.fastest_driver || 'Analyzing'}</div>
                                    <div class="metric-label">Fastest Driver</div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="metric-display status-good interactive-stat" data-tooltip="Most consistent performance throughout session">
                                    <div class="metric-value"><i class="fas fa-medal me-2"></i>${analysis.most_consistent_driver || 'Analyzing'}</div>
                                    <div class="metric-label">Most Consistent</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    if (comparisonData.driver_statistics) {
        html += `
            <div class="col-12 mb-3">
                <div class="driver-vs-indicator">
                    <i class="fas fa-fighters me-2"></i>DRIVER STATISTICS
                </div>
            </div>
        `;
        
        const drivers = Object.keys(comparisonData.driver_statistics);
        const driverStats = Object.values(comparisonData.driver_statistics);
        
        // Find best performance for comparison
        const fastestTimes = driverStats.map(stats => stats.fastest_lap_time || 999);
        const bestTime = Math.min(...fastestTimes);
        
        Object.entries(comparisonData.driver_statistics).forEach(([driver, stats], index) => {
            const fastestLap = stats.fastest_lap_time || 0;
            const averageLap = stats.average_lap_time || 0;
            const validLaps = stats.valid_laps || 0;
            
            // Performance calculations
            const timeGap = fastestLap > 0 && bestTime > 0 ? ((fastestLap - bestTime)).toFixed(3) : 0;
            const consistency = averageLap > 0 && fastestLap > 0 ? (100 - ((averageLap - fastestLap) / fastestLap * 100)).toFixed(1) : 0;
            const performance = fastestLap > 0 ? Math.max(0, 100 - (timeGap * 10)).toFixed(1) : 0;
            
            // Determine card styling based on performance
            let cardClass = 'driver-comparison-card';
            if (fastestLap === bestTime) cardClass = 'sector-card'; // Winner
            else if (timeGap < 0.5) cardClass = 'track-info-card'; // Close
            else cardClass = 'speed-analysis-card'; // Behind
            
            html += `
                <div class="col-lg-4 col-md-6 mb-4 slide-in-right">
                    <div class="data-card ${cardClass}">
                        <div class="result-card-header">
                            <i class="fas fa-user-racing me-2"></i>${driver}
                            ${fastestLap === bestTime ? '<i class="fas fa-crown ms-2 text-warning"></i>' : ''}
                        </div>
                        <div class="mt-3">
                            <div class="metric-display interactive-stat" data-tooltip="Best lap time achieved in session">
                                <div class="metric-value">${formatLapTime(fastestLap)}</div>
                                <div class="metric-label">Fastest Lap</div>
                            </div>
                            
                            <div class="row mt-3">
                                <div class="col-6">
                                    <div class="metric-display interactive-stat" data-tooltip="Average lap time across all valid laps">
                                        <div class="metric-value">${formatLapTime(averageLap)}</div>
                                        <div class="metric-label">Average</div>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="metric-display interactive-stat" data-tooltip="Number of completed laps">
                                        <div class="metric-value">${validLaps}</div>
                                        <div class="metric-label">Valid Laps</div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mt-3">
                                <div class="progress-modern">
                                    <div class="progress-bar-modern" style="width: ${performance}%"></div>
                                </div>
                                <div class="text-center mt-2">
                                    <small>Performance Index: ${performance}%</small>
                                </div>
                            </div>
                            
                            <div class="mt-3">
                                <div class="row text-center">
                                    <div class="col-6">
                                        <div class="interactive-stat" data-tooltip="Gap to fastest lap time">
                                            <small class="text-muted">Gap to Leader</small>
                                            <div class="fw-bold ${timeGap > 0 ? 'text-danger' : 'text-success'}">
                                                ${timeGap > 0 ? '+' : ''}${timeGap}s
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="interactive-stat" data-tooltip="Consistency rating based on lap time variance">
                                            <small class="text-muted">Consistency</small>
                                            <div class="fw-bold">${consistency}%</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
    }

    html += '</div>';
    return html;
}

/**
 * Export Functions
 */

// Export results
function exportResults() {
    if (!currentResults) {
        alert('No results to export. Please run an analysis first.');
        return;
    }

    const params = getFormParameters();
    const filename = `f1_analysis_${currentAnalysisType}_${params.year}_${params.grand_prix}_${params.session}.json`;

    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});

    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = filename;
    link.click();
}

/**
 * Initialization
 */

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('F1 Analytics Dashboard initialized');

    // Set up periodic refresh for real-time data (if needed)
    // This could be implemented later for true real-time updates
});
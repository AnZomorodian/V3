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
                                            <div class="metric-label">Overall Consistency CV</div>
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
                                <h5><i class="fas fa-chart-bar me-2"></i>Consistency Analysis</h5>
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

/**
 * Analysis History Functions
 */

// Load analysis history
async function loadAnalysisHistory() {
    try {
        const response = await fetch('/api/analysis-history');
        const history = await response.json();
        
        displayAnalysisHistory(history);
    } catch (error) {
        displayError('Failed to load analysis history: ' + error.message, 'historyContent');
    }
}

// Display analysis history
function displayAnalysisHistory(history) {
    const container = document.getElementById('historyContent');
    
    if (!history || history.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted">
                <i class="fas fa-history fa-2x mb-2"></i>
                <p>No analysis history available</p>
            </div>
        `;
        return;
    }
    
    let html = `
        <div class="table-responsive">
            <table class="table data-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Year</th>
                        <th>Grand Prix</th>
                        <th>Session</th>
                        <th>Analysis Type</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    history.forEach(item => {
        const date = new Date(item.created_at).toLocaleString();
        html += `
            <tr>
                <td>${date}</td>
                <td>${item.year}</td>
                <td>${item.grand_prix}</td>
                <td>${item.session}</td>
                <td>
                    <span class="badge bg-secondary">${item.analysis_type}</span>
                </td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = html;
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
    
    // Load initial analysis history
    loadAnalysisHistory();
    
    // Set up periodic refresh for real-time data (if needed)
    // This could be implemented later for true real-time updates
});

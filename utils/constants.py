"""
Constants for F1 Analytics application
"""

# Team colors for visualization
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#F91536',
    'Mercedes': '#6CD3BF',
    'McLaren': '#F58020',
    'Aston Martin': '#358C75',
    'Alpine': '#2293D1',
    'Williams': '#37003C',
    'AlphaTauri': '#5E8FAA',
    'Alfa Romeo': '#B12039',
    'Haas': '#B6BABD',
    'RB': '#6692FF',
    'Stake': '#52C832'
}

# Tire compound colors
TIRE_COLORS = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFFF33', 
    'HARD': '#FFFFFF',
    'INTERMEDIATE': '#33FF33',
    'WET': '#3333FF'
}

# Session types
SESSION_TYPES = [
    'Practice 1',
    'Practice 2', 
    'Practice 3',
    'Qualifying',
    'Sprint',
    'Sprint Qualifying',
    'Race'
]

# Driver abbreviations to full names mapping (2025 Season)
DRIVER_MAPPING = {
    'VER': 'Max Verstappen',
    'PER': 'Sergio Perez',
    'HAM': 'Lewis Hamilton',
    'RUS': 'George Russell',  
    'LEC': 'Charles Leclerc',
    'SAI': 'Carlos Sainz',
    'NOR': 'Lando Norris',
    'PIA': 'Oscar Piastri',
    'ALO': 'Fernando Alonso',
    'STR': 'Lance Stroll',
    'ALB': 'Alexander Albon',
    'COL': 'Franco Colapinto',
    'TSU': 'Yuki Tsunoda',
    'LAW': 'Liam Lawson',
    'BOT': 'Valtteri Bottas',
    'ZHO': 'Guanyu Zhou',
    'MAG': 'Kevin Magnussen',
    'HUL': 'Nico Hulkenberg',
    'GAS': 'Pierre Gasly',
    'OCO': 'Esteban Ocon',
    'BEA': 'Oliver Bearman',
    'HAD': 'Jack Doohan'
}

# Grand Prix locations
GRAND_PRIX_LOCATIONS = {
    'Bahrain': 'Bahrain International Circuit',
    'Saudi Arabia': 'Jeddah Corniche Circuit',
    'Australia': 'Albert Park Circuit',
    'Japan': 'Suzuka International Racing Course',
    'China': 'Shanghai International Circuit',
    'Miami': 'Miami International Autodrome',
    'Emilia Romagna': 'Autodromo Enzo e Dino Ferrari',
    'Monaco': 'Circuit de Monaco',
    'Canada': 'Circuit Gilles Villeneuve',
    'Spain': 'Circuit de Barcelona-Catalunya',
    'Austria': 'Red Bull Ring',
    'Great Britain': 'Silverstone Circuit',
    'Hungary': 'Hungaroring',
    'Belgium': 'Circuit de Spa-Francorchamps',
    'Netherlands': 'Circuit Zandvoort',
    'Italy': 'Autodromo Nazionale di Monza',
    'Singapore': 'Marina Bay Street Circuit',
    'United States': 'Circuit of the Americas',
    'Mexico': 'Autodromo Hermanos Rodriguez',
    'Brazil': 'Interlagos Circuit',
    'Las Vegas': 'Las Vegas Street Circuit',
    'Qatar': 'Losail International Circuit',
    'Abu Dhabi': 'Yas Marina Circuit'
}

# Driver to team mappings for 2024 season
DRIVER_TEAMS = {
    'VER': 'Red Bull Racing',
    'PER': 'Red Bull Racing',
    'LEC': 'Ferrari',
    'SAI': 'Ferrari',
    'HAM': 'Mercedes',
    'RUS': 'Mercedes',
    'NOR': 'McLaren',
    'PIA': 'McLaren',
    'ALO': 'Aston Martin',
    'STR': 'Aston Martin',
    'GAS': 'Alpine',
    'OCO': 'Alpine',
    'ALB': 'Williams',
    'SAR': 'Williams',
    'TSU': 'AlphaTauri',
    'RIC': 'AlphaTauri',
    'BOT': 'Alfa Romeo',
    'ZHO': 'Alfa Romeo',
    'MAG': 'Haas',
    'HUL': 'Haas',
    # Additional drivers that might appear
    'DEV': 'Red Bull Racing',
    'LAW': 'AlphaTauri',
    'BEA': 'Ferrari',
    'COL': 'Williams'
}
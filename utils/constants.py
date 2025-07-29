"""
Constants for F1 Analytics application
"""

# Team colors for visualization
TEAM_COLORS = {
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

# Driver abbreviations to full names mapping
DRIVER_MAPPING = {
    'VER': 'Max Verstappen',
    'HAM': 'Lewis Hamilton',
    'LEC': 'Charles Leclerc',
    'SAI': 'Carlos Sainz',
    'RUS': 'George Russell',
    'NOR': 'Lando Norris',
    'PIA': 'Oscar Piastri',
    'ALO': 'Fernando Alonso',
    'STR': 'Lance Stroll',
    'ALB': 'Alexander Albon',
    'SAR': 'Logan Sargeant',
    'TSU': 'Yuki Tsunoda',
    'RIC': 'Daniel Ricciardo',
    'BOT': 'Valtteri Bottas',
    'ZHO': 'Guanyu Zhou',
    'MAG': 'Kevin Magnussen',
    'HUL': 'Nico Hulkenberg',
    'GAS': 'Pierre Gasly',
    'OCO': 'Esteban Ocon',
    'PER': 'Sergio Perez'
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
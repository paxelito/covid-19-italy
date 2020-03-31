import os
from datetime import date, timedelta

COVID_DATA_FOLDER = os.path.abspath(os.path.join('/Users/afilisetti/Documents/git-projects/COVID-19/'))
CITIES_DATA_JSON_URI = os.path.abspath(os.path.join(COVID_DATA_FOLDER, "dati-json/dpc-covid19-ita-province.json"))
REGIONS_DATA_JSON_URI = os.path.abspath(os.path.join(COVID_DATA_FOLDER, "dati-json/dpc-covid19-ita-regioni.json"))
ITALY_MAP = fp = os.path.abspath(os.path.join(COVID_DATA_FOLDER, "aree/shp/dpc-covid19-ita-aree.shp"))

AVG_GROWTH_RATE_WINDOW = 7

NW_REGIONS = ["Lombardia", "Piemonte", "Valle d'Aosta", "Liguria"]
NE_REGIONS = ["Veneto", "Emilia-Romagna", "Friuli Venezia Giulia", "P.A. Bolzano", "P.A. Trento"]
C_REGIONS = ["Lazio", "Toscana", "Umbria", "Marche"]
S_REGIONS = ["Campania", "Abruzzo", "Molise", "Campania", "Puglia", "Basilicata"]
ISLANDS = ["Sicilia", "Sardegna"]

CITIES_LOMBARDIA = ['Bergamo', 'Brescia', 'Milano', 'Lodi', 'Monza e della Brianza', 'Sondrio', 'Varese', 'Lecco', 'Pavia', 'Cremona', 'Como', 'Mantova']
CITIES_EMILIA = ['Piacenza', "Reggio nell'Emilia", 'Ravenna', 'Ferrara', 'Modena', 'Forlì-Cesena', 'Bologna', 'Parma', 'Rimini']
CITIES_VENETO = ['Padova', 'Rovigo', 'Treviso', 'Verona', 'Belluno', 'Vicenza', 'Venezia']
CITIES_CAMPANIA = ['Napoli', 'Benevento', 'Avellino', 'Salerno', 'Caserta']


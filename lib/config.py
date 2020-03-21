AVG_GROWTH_RATE_WINDOW = 7

NW_REGIONS = ["Lombardia", "Piemonte", "Valle d'Aosta", "Liguria"]
NE_REGIONS = ["Emilia Romagna", "Veneto", "Friuli Venezia Giulia", "P.A. Bolzano", "P.A. Trento"]
C_REGIONS = ["Lazio", "Toscana", "Umbria", "Marche"]
S_REGIONS = ["Campania", "Abruzzo", "Molise", "Campania", "Puglia", "Basilicata"]
ISLANDS = ["Sicilia", "Sardegna"]

CITIES_LOMBARDIA = get_cities_from_regions("Lombardia")
CITIES_EMILIA = get_cities_from_regions("Emilia")
CITIES_VENETO = get_cities_from_regions("Veneto")
CITIES_CAMPANIA = get_cities_from_regions("Campania")

today = date.today() - timedelta(days=1)  # ).strftime('%Y-%m-%d')
print(f"Date of the analysis: {today.strftime('%Y-%m-%d')}")

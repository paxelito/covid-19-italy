from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import seaborn as sns
import geopandas as gpd
import numpy as np
from src.config import CITIES_DATA_JSON_URI, REGIONS_DATA_JSON_URI, ITALY_MAP, AVG_GROWTH_RATE_WINDOW


class ItalianCovidData:
    """
    Class containing all the methods to compute analysis
    """

    def __init__(self, avg_growth_rate_window=AVG_GROWTH_RATE_WINDOW):
        # LOAD POPULATION PER CITIES
        self.population = pd.read_csv("../data/province2.csv", sep=",", header=0, index_col='Provincia')
        self.population = self.population.loc[self.population['Età'] == 'Totale']

        # LOAD REGIONAL POPULATION
        self.regional_population = pd.read_csv("../data/regioni.tsv", sep="\t", header=0, index_col='regione')

        # CITIES
        self.cities_data_json = pd.read_json(CITIES_DATA_JSON_URI)
        self.cities_data_json["data"] = pd.to_datetime(self.cities_data_json["data"])
        self.cities_data_json = self.cities_data_json[self.cities_data_json.denominazione_provincia != 'In fase di definizione/aggiornamento']

        # REGIONS
        self.regions_data_json = pd.read_json(REGIONS_DATA_JSON_URI)
        self.regions_data_json = pd.merge(
            left=self.regions_data_json,
            right=self.regional_population,
            how='inner',
            left_on='denominazione_regione',
            right_on=self.regional_population.index
        )
        self.map_df = gpd.read_file(ITALY_MAP)
        self.regions_data_json["data"] = pd.to_datetime(self.regions_data_json["data"])
        self.regions_data_json["ratio_positivi"] = self.regions_data_json["nuovi_positivi"] / self.regions_data_json["tamponi"]
        self.regions_data_json["fatality"] = self.regions_data_json["deceduti"] / self.regions_data_json["totale_casi"]
        self.regions_data_json["mortalityX1000"] = self.regions_data_json["deceduti"] / self.regions_data_json["popolazione"] * 1000

        self.today = date.today()

    def data_summary(self):
        print(f"--- Latest Update: {self.today} ---\n")
        print("\n --- REGIONS DATASET --- \n")
        print(self.regions_data_json.info())
        print("\n --- CITIES DATASET --- \n")
        print(self.cities_data_json.info())
        print("\n --- METADATA --- \n")
        print(self.population.info())
        print(self.regional_population.info())

    def show_map_cases(self, current_date=None):
        filter_time = self.today.strftime('%Y-%m-%d') if not current_date else current_date
        filter_time_7 = str(date.today() - timedelta(days=7))
        filter_time_15 = str(date.today() - timedelta(days=15))
        today_data_json = self.cities_data_json.loc[(pd.to_datetime(self.cities_data_json['data']).dt.strftime('%Y-%m-%d') == filter_time) &
                                                    (self.cities_data_json['sigla_provincia'] != "")]
        today_data_json_7 = self.cities_data_json.loc[(pd.to_datetime(self.cities_data_json['data']).dt.strftime('%Y-%m-%d') == filter_time_7) &
                                                      (self.cities_data_json['sigla_provincia'] != "")]
        today_data_json_15 = self.cities_data_json.loc[(pd.to_datetime(self.cities_data_json['data']).dt.strftime('%Y-%m-%d') == filter_time_15) &
                                                       (self.cities_data_json['sigla_provincia'] != "")]

        data_temp = pd.merge(today_data_json, today_data_json_7[['codice_provincia', 'totale_casi']], on='codice_provincia')
        data = pd.merge(data_temp, today_data_json_15[['codice_provincia', 'totale_casi']], on='codice_provincia')

        data['growth_factor'] = (data['totale_casi_x'] - data['totale_casi_y']) / (data['totale_casi_y'] - data['totale_casi'])

        ax = self.map_df.plot(figsize=(20, 20), alpha=0.4, edgecolor='k')
        today_data_json.plot(kind="scatter",
                             x="long",
                             y="lat",
                             alpha=0.5,
                             s=data["growth_factor"] * 100,
                             label="Growth Factor", figsize=(10, 7),
                             c=data["growth_factor"],
                             cmap=plt.get_cmap("jet"),
                             colorbar=True,
                             ax=ax)

    def plot_region(self, region):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        ax = sns.lineplot(x="data",
                          y="totale_casi",
                          hue="sigla_provincia",
                          linestyle='dotted',
                          marker="o",
                          data=self.cities_data_json.query(f"denominazione_regione == '{region}'"),
                          )
        plt.subplot(1, 2, 2)
        ax.set_ylabel("Total Cases")
        bx = sns.lineplot(x="data",
                          y="totale_casi",
                          hue="sigla_provincia",
                          linestyle='dotted',
                          marker="o",
                          data=self.cities_data_json.query(f"denominazione_regione == '{region}'"),
                          )
        bx.set_yscale('log')
        bx.set_ylabel("log(Total Cases)")
        plt.grid(True, which="both", ls="--", c='gray')
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        bx.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        plt.draw()

    def plot_region_indicators(self, regions_area):
        self._plot_regions(data=self.cities_data_json,
                           data_filter=regions_area,
                           y='totale_casi')
        vars_of_interest = ['totale_casi', 'totale_positivi', 'deceduti', 'terapia_intensiva', 'tamponi', 'fatality', 'mortalityX1000']
        for var_of_interest in vars_of_interest:
            self._plot_regions(data=self.regions_data_json,
                               data_filter=regions_area,
                               y=var_of_interest)

    @staticmethod
    def _plot_regions(data: pd.DataFrame, data_filter: str, y: str = 'totale_casi') -> None:

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        ax = sns.lineplot(x="data",
                          y=y,
                          hue="denominazione_regione",
                          data=data.query(f"denominazione_regione in {data_filter}"),
                          linestyle='dotted',
                          marker="o"
                          )
        plt.subplot(1, 2, 2)
        bx = sns.lineplot(x="data",
                          y=y,
                          hue="denominazione_regione",
                          markers=True,
                          dashes=True,
                          data=data.query(f"denominazione_regione in {data_filter}"),
                          linestyle='dotted',
                          marker="o"
                          )
        bx.set_yscale("log")
        bx.set_ylabel(f"log({y})")
        plt.grid(True, which="both")
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        bx.xaxis.set_major_formatter(DateFormatter('%m-%d'))
        plt.show()

    def growth_rates(self, areas, regions=True, area_target='denominazione_provincia', indicator='totale_casi', grw=7):
        data = self.regions_data_json if regions else self.cities_data_json
        growth_rates = dict()
        for area in areas:
            data_area = data.loc[data[area_target] == area]
            growth_rate = list()
            growth_rates[area] = dict()
            for idx, i in enumerate(data_area[indicator]):
                if idx > 0:
                    if list(data_area[indicator])[idx - 1] > 1:
                        temp_gr = ((i / list(data_area[indicator])[idx - 1]) - 1) * 100
                        growth_rate.append((temp_gr, list(data_area['data'])[idx]))

            growth_rates[area]['growth_rate'] = growth_rate
            growth_rate_n, growth_rate_date = zip(*growth_rates[area]['growth_rate'])

            avg_gr = list()
            for idx, g in enumerate(growth_rate_n[0:len(growth_rate_n) - grw + 1]):
                avg_gr.append(np.mean(growth_rate_n[idx:idx + grw]))

            growth_rates[area]['avg_growth_rate'] = avg_gr

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        for area in areas:
            growth_rate_n, growth_rate_date = zip(*growth_rates[area]['growth_rate'])
            plt.plot(
                growth_rate_date[:],
                growth_rate_n[:],
                linestyle='solid',
                marker="o"
            )
            plt.xlabel("Date (data starts from case 0)")
            plt.ylabel("Growth Rate (%)")
            plt.title(f"Daily Growth Rate ({indicator})")
        plt.legend(areas)
        plt.xticks(rotation=90)
        plt.subplot(1, 2, 2)
        for area in areas:
            plt.plot(growth_rates[area]['avg_growth_rate'][:], linestyle='solid', marker="o")
            plt.xlabel(f"Average growth Factor (Floating window: {grw} days) from case 0")
            plt.ylabel("AVG Growth Factor (%)")
            plt.title(f"AVG Daily Growth Factor ({indicator})")
        plt.legend(areas)
        plt.draw()

    def growth_factors(self, areas, regions=True, area_target='denominazione_provincia', indicator='totale_casi', grw=(7, )):
        data = self.regions_data_json if regions else self.cities_data_json
        growth_factors = dict()
        for area in areas:
            data_area = data.loc[data[area_target] == area]
            growth_factors[area] = dict()
            growth_factors[area]['growth_rate'] = dict()
            for g in grw:
                growth_rate = list()
                for idx, i in enumerate(data_area[indicator]):
                    if idx > (2 * g):
                        if list(data_area[indicator])[idx - (2 * g)] > 1:
                            delta_t1 = (i - list(data_area[indicator])[idx - g])
                            delta_t2 = (list(data_area[indicator])[idx - g] - list(data_area[indicator])[idx - (2 * g)])
                            temp_gr = delta_t1 / delta_t2 if delta_t2 != 0 else 0
                            growth_rate.append((temp_gr, list(data_area['data'])[idx]))
                growth_factors[area]['growth_rate'][g] = growth_rate

        plt.figure(figsize=(10*len(grw), 10))
        for idx, g in enumerate(grw):
            plt.subplot(1, len(grw), idx+1)
            for area in areas:
                growth_rate_n, growth_rate_date = zip(*growth_factors[area]['growth_rate'][g])

                plt.plot(
                    growth_rate_date[:],
                    growth_rate_n[:],
                    linestyle='solid' if np.mean(growth_rate_n[-3:]) >= 1 else ':',
                    marker="."
                )
                plt.xlabel(f"Date (data start {2*g} days after case 0)")
                plt.ylabel("Growth Factor")
                plt.title(f"$(X_t-X_{{t-{g}}})/(X_{{t-{g}}}-X_{{t-{2*g}}})$")
            plt.suptitle(f"Growth Factor ({indicator})")
            plt.legend(areas)
            plt.axhline(y=1, linewidth=4, color='r')
            plt.xticks(rotation=90)
        plt.draw()

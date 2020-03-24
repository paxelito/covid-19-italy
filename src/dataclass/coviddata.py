from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np
from src.config import CITIES_DATA_JSON_URI, REGIONS_DATA_JSON_URI, ITALY_MAP, AVG_GROWTH_RATE_WINDOW


class ItalianCovidData:

    def __init__(self, avg_growth_rate_window=AVG_GROWTH_RATE_WINDOW):
        self.cities_data_json = pd.read_json(CITIES_DATA_JSON_URI)
        self.cities_data_json["data"] = pd.to_datetime(self.cities_data_json["data"]).dt.strftime('%Y-%m-%d')
        self.regions_data_json = pd.read_json(REGIONS_DATA_JSON_URI)
        self.map_df = gpd.read_file(ITALY_MAP)

        self.regions_data_json["data"] = pd.to_datetime(self.regions_data_json["data"]).dt.strftime('%Y-%m-%d')
        self.regions_data_json["ratio_positivi"] = self.regions_data_json["nuovi_attualmente_positivi"] / self.regions_data_json["tamponi"]
        self.regions_data_json["mortality"] = self.regions_data_json["deceduti"] / self.regions_data_json["totale_casi"]

        self.today = date.today()
        self.yesterday = date.today() - timedelta(days=1)

    def data_summary(self):
        print(f"--- Latest Update: {self.today} ---\n")
        print("\n --- REGIONS DATASET --- \n")
        print(self.regions_data_json.info())
        print("\n --- CITIES DATASET --- \n")
        print(self.cities_data_json.info())

    def show_map_cases(self, today=True):
        filter_time = self.today.strftime('%Y-%m-%d') if today else self.yesterday.strftime('%Y-%m-%d')
        today_data_json = self.cities_data_json.loc[(self.cities_data_json['data'] == filter_time) & (self.cities_data_json['sigla_provincia'] != "")]
        ax = self.map_df.plot(figsize=(10, 10), alpha=0.4, edgecolor='k')
        today_data_json.plot(kind="scatter",
                             x="long",
                             y="lat",
                             alpha=0.5,
                             s=today_data_json["totale_casi"] / 5,
                             label="Totale Casi", figsize=(10, 7),
                             c=today_data_json["totale_casi"],
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
        plt.draw()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        bx.set_xticklabels(bx.get_xticklabels(), rotation=90)

    def plot_region_indicators(self, regions_area):
        self._plot_regions(self.cities_data_json, regions_area, 'totale_casi')
        vars_of_interest = ['totale_casi', 'deceduti', 'terapia_intensiva', 'tamponi']#, 'ratio_positivi', 'mortality']
        for var_of_interest in vars_of_interest:
            self._plot_regions(data=self.regions_data_json,
                               data_filter=regions_area,
                               y=var_of_interest)

    @staticmethod
    def _plot_regions(data, data_filter, y='totale_casi'):
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
        plt.draw()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        bx.set_xticklabels(bx.get_xticklabels(), rotation=90)

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
                        temp_gr = i / list(data_area[indicator])[idx - 1]
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
            plt.ylabel("Growth Rate")
            plt.title(f"Daily Growth Rate ({indicator})")
        plt.legend(areas)
        plt.xticks(rotation=90)
        plt.subplot(1, 2, 2)
        for area in areas:
            plt.plot(growth_rates[area]['avg_growth_rate'][:], linestyle='solid', marker="o")
            plt.xlabel(f"Average growth Factor (Floating window: {grw} days) from case 0")
            plt.ylabel("AVG Growth Factor")
            plt.title(f"AVG Daily Growth Factor ({indicator})")
        plt.legend(areas)
        plt.draw()

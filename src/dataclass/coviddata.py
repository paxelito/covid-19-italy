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
        self.population['totale'] = self.population['Totale Maschi'] + self.population['Totale Femmine']
        self.population.reset_index(inplace=True)

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
        self.regions_data_json["ratio_positivi"] = self.regions_data_json["totale_casi"] / self.regions_data_json["tamponi"]
        self.regions_data_json["fatality"] = self.regions_data_json["deceduti"] / self.regions_data_json["totale_casi"]
        self.regions_data_json["mortalityX1000"] = self.regions_data_json["deceduti"] / self.regions_data_json["popolazione"] * 1000
        self.today = date.today()
        self.set_cities_data_today()

    def set_cities_data_today(self, day_window=7, days_before=0):
        # Set the different filter times
        filter_times = [str(date.today() - timedelta(days=(day_window * x + days_before))) for x in range(0, 4)]

        ''' Procedure 2 under investigation 
        cities_data_df = [self.cities_data_json.loc[(pd.to_datetime(self.cities_data_json['data']).dt.strftime('%Y-%m-%d') == x) &
                                                    (self.cities_data_json['sigla_provincia'] != "")][['codice_provincia', 'totale_casi']].rename(columns={
            "totale_casi": f"totale_casi_{idx}"}) for idx, x in enumerate(filter_times)]

        merge = partial(pd.merge, on=['codice_provincia'], how='inner')
        cities_data_today_2 = reduce(merge, cities_data_df)
        '''

        # Filter global data for times
        today_data_json = self.cities_data_json.loc[(pd.to_datetime(self.cities_data_json['data']).dt.strftime('%Y-%m-%d') == filter_times[0]) &
                                                    (self.cities_data_json['sigla_provincia'] != "")]
        today_data_json_7 = self.cities_data_json.loc[(pd.to_datetime(self.cities_data_json['data']).dt.strftime('%Y-%m-%d') == filter_times[1]) &
                                                      (self.cities_data_json['sigla_provincia'] != "")]
        today_data_json_15 = self.cities_data_json.loc[(pd.to_datetime(self.cities_data_json['data']).dt.strftime('%Y-%m-%d') == filter_times[2]) &
                                                       (self.cities_data_json['sigla_provincia'] != "")]
        today_data_json_22 = self.cities_data_json.loc[(pd.to_datetime(self.cities_data_json['data']).dt.strftime('%Y-%m-%d') == filter_times[3]) &
                                                       (self.cities_data_json['sigla_provincia'] != "")]

        # Merge together cities data
        cities_data_today = pd.merge(today_data_json, today_data_json_7[['codice_provincia', 'totale_casi']], on='codice_provincia', suffixes=("_0", f"_{day_window}"))
        cities_data_today = pd.merge(cities_data_today, today_data_json_15[['codice_provincia', 'totale_casi']], on='codice_provincia').rename(
            columns={'totale_casi': f'totale_casi_{day_window * 2}'})
        cities_data_today = pd.merge(cities_data_today, today_data_json_22[['codice_provincia', 'totale_casi']], on='codice_provincia').rename(
            columns={'totale_casi': f'totale_casi_{day_window * 3}'})

        # filter province data on data
        today_province_json = self.regions_data_json.loc[(pd.to_datetime(self.regions_data_json['data']).dt.strftime('%Y-%m-%d') == filter_times[0])]
        today_province_json_7 = self.regions_data_json.loc[(pd.to_datetime(self.regions_data_json['data']).dt.strftime('%Y-%m-%d') == filter_times[1])]
        today_province_json_15 = self.regions_data_json.loc[(pd.to_datetime(self.regions_data_json['data']).dt.strftime('%Y-%m-%d') == filter_times[2])]
        today_province_json_22 = self.regions_data_json.loc[(pd.to_datetime(self.regions_data_json['data']).dt.strftime('%Y-%m-%d') == filter_times[3])]

        # merge together province data
        province_data_today = pd.merge(today_province_json, today_province_json_7[['codice_regione', 'tamponi', 'popolazione']], on=['codice_regione', 'popolazione'],
                                       suffixes=("_0", f"_{day_window}"))
        province_data_today = pd.merge(province_data_today, today_province_json_15[['codice_regione', 'tamponi', 'popolazione']], on=['codice_regione', 'popolazione']).rename(
            columns={'tamponi': f'tamponi_{day_window * 2}'})
        province_data_today = pd.merge(province_data_today, today_province_json_22[['codice_regione', 'tamponi', 'popolazione']], on=['codice_regione', 'popolazione']).rename(
            columns={'tamponi': f'tamponi_{day_window * 3}'})

        # merge province data into cities data
        cities_data_today = pd.merge(cities_data_today, province_data_today[['codice_regione',
                                                                             'popolazione',
                                                                             f'tamponi_{day_window * 0}',
                                                                             f'tamponi_{day_window * 1}',
                                                                             f'tamponi_{day_window * 2}',
                                                                             f'tamponi_{day_window * 3}']],
                                     left_on='codice_regione',
                                     right_on='codice_regione')

        # Merge province data into cities
        self.cities_data_today = cities_data_today.merge(self.population[['Provincia', 'Totale Maschi', 'Totale Femmine', 'totale']],
                                                         left_on="denominazione_provincia",
                                                         right_on="Provincia"
                                                         )

        # ASSUMPTION Adjust tamponi proportionally to region population
        self.cities_data_today[f'tamponi_{day_window * 0}'] = self.cities_data_today[f'tamponi_{day_window * 0}'] / \
                                                              self.cities_data_today['popolazione'] * self.cities_data_today['totale']
        self.cities_data_today[f'tamponi_{day_window * 1}'] = self.cities_data_today[f'tamponi_{day_window * 1}'] / \
                                                              self.cities_data_today['popolazione'] * self.cities_data_today['totale']
        self.cities_data_today[f'tamponi_{day_window * 2}'] = self.cities_data_today[f'tamponi_{day_window * 2}'] / \
                                                              self.cities_data_today['popolazione'] * self.cities_data_today['totale']
        self.cities_data_today[f'tamponi_{day_window * 3}'] = self.cities_data_today[f'tamponi_{day_window * 3}'] / \
                                                              self.cities_data_today['popolazione'] * self.cities_data_today['totale']

        # Compute growth factors
        self.cities_data_today['growth_factor'] = (self.cities_data_today[f'totale_casi_{day_window * 0}'] - self.cities_data_today[f'totale_casi_{day_window * 1}']) / \
                                                  (self.cities_data_today[f'totale_casi_{day_window * 1}'] - self.cities_data_today[f'totale_casi_{day_window * 2}'])

        self.cities_data_today['growth_factor_prec'] = (self.cities_data_today[f'totale_casi_{day_window * 1}'] - self.cities_data_today[f'totale_casi_{day_window * 2}']) / \
                                                       (self.cities_data_today[f'totale_casi_{day_window * 2}'] - self.cities_data_today[f'totale_casi_{day_window * 3}'])

        # Compute normalized growth factors
        self.cities_data_today['growth_factor_norm'] = ((self.cities_data_today[f'totale_casi_{day_window * 0}'] - self.cities_data_today[f'totale_casi_{day_window * 1}']) /
                                                        (self.cities_data_today[f'tamponi_{day_window * 0}'] - self.cities_data_today[f'tamponi_{day_window * 1}'])) / \
                                                       ((self.cities_data_today[f'totale_casi_{day_window * 1}'] - self.cities_data_today[f'totale_casi_{day_window * 2}']) /
                                                        (self.cities_data_today[f'tamponi_{day_window * 1}'] - self.cities_data_today[f'tamponi_{day_window * 2}']))

        self.cities_data_today['growth_factor_prec_norm'] = ((self.cities_data_today[f'totale_casi_{day_window * 1}'] - self.cities_data_today[f'totale_casi_{day_window * 2}']) /
                                                             (self.cities_data_today[f'tamponi_{day_window * 1}'] - self.cities_data_today[f'tamponi_{day_window * 2}'])) / \
                                                            ((self.cities_data_today[f'totale_casi_{day_window * 2}'] - self.cities_data_today[f'totale_casi_{day_window * 3}']) /
                                                             (self.cities_data_today[f'tamponi_{day_window * 2}'] - self.cities_data_today[f'tamponi_{day_window * 3}']))

        self.cities_data_today['incidence'] = self.cities_data_today[f'totale_casi_{day_window * 0}'] / self.cities_data_today['totale'] * 1000
        self.cities_data_today['incidence_prec'] = self.cities_data_today[f'totale_casi_{day_window * 1}'] / self.cities_data_today['totale'] * 1000

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

        ax = self.map_df.plot(figsize=(20, 20), alpha=0.4, edgecolor='k')
        self.cities_data_today.plot(kind="scatter",
                                    x="long",
                                    y="lat",
                                    alpha=0.5,
                                    s=self.cities_data_today["growth_factor"] * 100,
                                    label="Growth Factor", figsize=(10, 7),
                                    c=self.cities_data_today["growth_factor"],
                                    cmap=plt.get_cmap("jet"),
                                    colorbar=True,
                                    ax=ax)

    def scatter_gf_incidence(self, regioni=None, norm=True):
        data_temp = self.cities_data_today.loc[self.cities_data_today['denominazione_regione'].isin(regioni)] if regioni else self.cities_data_today

        norm = '_norm' if norm else ''

        data = data_temp.rename(columns={
            "denominazione_regione": "regione",
            "totale": "popolazione_province"})

        plt.figure(figsize=(20, 20))

        ax = sns.scatterplot(x='incidence',
                             y=f'growth_factor{norm}',
                             hue='regione',
                             data=data,
                             alpha=0.8,
                             size='totale_casi_0',
                             sizes=(20, 500))

        ax.axhline(np.mean(list(data[f'growth_factor{norm}'][~np.isinf(data[f'growth_factor{norm}'])])), linestyle=":")
        ax.axhline(1, color="r", linestyle=":")
        ax.axvline(np.mean(list(data['incidence'])), linestyle=":")
        ax.set_xlabel("Contagious X 1000 inhabitants")
        ax.set_ylabel(f"Growth factor{norm} ($(X_t-X_{{7}})/(X_{{7}}-X_{{15}})$)")
        ax.set_title(f"Growth Factor{norm} vs Incidence")

        ax.annotate("Inflection Point",
                    (max(list(data['incidence'])) * 0.8, 1),
                    color='red',
                    fontsize=15)
        ax.annotate("avg(incidence)",
                    (np.mean(list(data['incidence'])), max(list(data[f'growth_factor{norm}'])) * 0.9),
                    color='blue',
                    fontsize=15
                    )
        ax.annotate(f"avg(growth factor{norm})",
                    (max(list(data['incidence'])) * 0.8, np.mean(list(data[f'growth_factor{norm}']))),
                    color='blue',
                    fontsize=15
                    )

        for i, txt in enumerate(list(data['Provincia'])):
            ax.annotate(txt, (list(data['incidence'])[i], list(data[f'growth_factor{norm}'])[i]))
            color = 'green' if list(data[f'growth_factor{norm}'])[i] < list(data[f'growth_factor_prec{norm}'])[i] else "red"
            ax.annotate('',
                        xy=(list(data['incidence'])[i], list(data[f'growth_factor{norm}'])[i]),
                        xytext=(list(data['incidence_prec'])[i], list(data[f'growth_factor_prec{norm}'])[i]),
                        arrowprops={'arrowstyle': '->', 'lw': 1, 'color': color, 'ls': "dotted"},
                        va='center')

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
        vars_of_interest = ['totale_casi', 'totale_positivi', 'ratio_positivi', 'deceduti', 'terapia_intensiva', 'totale_ospedalizzati',
                            'isolamento_domiciliare', 'dimessi_guariti', 'tamponi', 'fatality', 'mortalityX1000']
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

    def growth_factors(self, areas, regions=True, area_target='denominazione_provincia', indicator='totale_casi', grw=(7,)):
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

        plt.figure(figsize=(10 * len(grw), 10))
        for idx, g in enumerate(grw):
            plt.subplot(1, len(grw), idx + 1)
            for area in areas:
                growth_rate_n, growth_rate_date = zip(*growth_factors[area]['growth_rate'][g])

                plt.plot(
                    growth_rate_date[:],
                    growth_rate_n[:],
                    linestyle='solid' if np.mean(growth_rate_n[-3:]) >= 1 else ':',
                    marker="."
                )
                plt.xlabel(f"Date (data start {2 * g} days after case 0)")
                plt.ylabel("Growth Factor")
                plt.title(f"$(X_t-X_{{t-{g}}})/(X_{{t-{g}}}-X_{{t-{2 * g}}})$")
            plt.suptitle(f"Growth Factor ({indicator})")
            plt.legend(areas)
            plt.axhline(y=1, linewidth=4, color='r')
            plt.xticks(rotation=90)
        plt.draw()

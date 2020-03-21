import matplotlib.pyplot as plt
import seaborn as sns


class ItalianCovidData:

    def __init__(self):
        pass

    def plot_region(self, region, data_json):
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        ax = sns.lineplot(x="data",
                          y="totale_casi",
                          hue="sigla_provincia",
                          linestyle='dotted',
                          marker="o",
                          data=data_json.query(f"denominazione_regione == '{region}'"),
                         )
        plt.subplot(1,2,2)
        ax.set_ylabel("Totale Cases")
        bx = sns.lineplot(x="data",
                          y="totale_casi",
                          hue="sigla_provincia",
                          linestyle='dotted',
                          marker="o",
                          data=data_json.query(f"denominazione_regione == '{region}'"),
                         )
        bx.set_yscale('log')
        bx.set_ylabel("log(Total Cases)")
        plt.draw()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        bx.set_xticklabels(bx.get_xticklabels(), rotation=90)


    def plot_regions(self, data, data_filter, y='totale_casi'):
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


    def plot_region_indicators(self, cities_data_json, regions_data_json, regions_area):
        plot_regions(cities_data_json, regions_area, 'totale_casi')
        plot_regions(regions_data_json, regions_area, 'totale_casi')
        plot_regions(regions_data_json, regions_area, 'deceduti')
        plot_regions(regions_data_json, regions_area, 'terapia_intensiva')
        plot_regions(regions_data_json, regions_area, 'tamponi')
        plot_regions(regions_data_json, regions_area, 'ratio_positivi')


    def growth_rate(self, city, avg_growth_rate_window=AVG_GROWTH_RATE_WINDOW):
        data_city = data_json.loc[data_json['sigla_provincia'] == city]
        growth_rate = list()
        for idx, i in enumerate(data_city['totale_casi']):
            if idx > 0:
                if list(data_city['totale_casi'])[idx - 1] > 0:
                    growth_rate.append(i / list(data_city['totale_casi'])[idx - 1])

        avg_gr = list()
        for idx, g in enumerate(growth_rate[0:len(growth_rate) - avg_growth_rate_window]):
            avg_gr.append(np.mean(growth_rate[idx:idx + avg_growth_rate_window]))

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        ax = plt.plot(growth_rate[2:], linestyle='dotted', marker="o")
        plt.xlabel("Days from 26/2");
        plt.ylabel("Growth rate (%)");
        plt.title(f"Daily Growth Rate ({city})");
        plt.subplot(1, 2, 2)
        plt.plot(avg_gr, linestyle='dotted', marker="o")
        plt.xlabel(f"Average growth rate (Floating window: {avg_growth_rate_window} days)");
        plt.ylabel("AVG Growth rate (%)");
        plt.title(f"AVG Daily Growth Rate ({city})");


    def growth_rate(self, city, avg_growth_rate_window=AVG_GROWTH_RATE_WINDOW):
        data_city = data_json.loc[data_json['sigla_provincia'] == city]
        growth_rate = list()
        for idx, i in enumerate(data_city['totale_casi']):
            if idx > 0:
                if list(data_city['totale_casi'])[idx - 1] > 0:
                    growth_rate.append(i / list(data_city['totale_casi'])[idx - 1])

        avg_gr = list()
        for idx, g in enumerate(growth_rate[0:len(growth_rate) - avg_growth_rate_window]):
            avg_gr.append(np.mean(growth_rate[idx:idx + avg_growth_rate_window]))

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        ax = plt.plot(growth_rate[2:], linestyle='dotted', marker="o")
        plt.xlabel("Days from 26/2");
        plt.ylabel("Growth rate (%)");
        plt.title(f"Daily Growth Rate ({city})");
        plt.subplot(1, 2, 2)
        plt.plot(avg_gr, linestyle='dotted', marker="o")
        plt.xlabel(f"Average growth rate (Floating window: {avg_growth_rate_window} days)");
        plt.ylabel("AVG Growth rate (%)");
        plt.title(f"AVG Daily Growth Rate ({city})");


    def growth_rates(self,
                     data,
                     areas,
                     area_target='sigla_provincia',
                     indicator='totale_casi',
                     avg_growth_rate_window=AVG_GROWTH_RATE_WINDOW):
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
            for idx, g in enumerate(growth_rate_n[0:len(growth_rate_n) - avg_growth_rate_window + 1]):
                avg_gr.append(np.mean(growth_rate_n[idx:idx + avg_growth_rate_window]))

            growth_rates[area]['avg_growth_rate'] = avg_gr

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        for area in areas:
            growth_rate_n, growth_rate_date = zip(*growth_rates[area]['growth_rate'])
            ax = plt.plot(
                growth_rate_date[:],
                growth_rate_n[:],
                linestyle='solid',
                marker="o"
            )
            plt.xlabel("Date (data starts from case 0)");
            plt.ylabel("Growth Rate");
            plt.title(f"Daily Growth Rate ({area})");
        plt.legend(areas)
        plt.xticks(rotation=90);
        plt.subplot(1, 2, 2)
        for area in areas:
            plt.plot(growth_rates[area]['avg_growth_rate'][:], linestyle='solid', marker="o")
            plt.xlabel(f"Average growth Factor (Floating window: {avg_growth_rate_window} days) from case 0");
            plt.ylabel("AVG Growth Factor");
            plt.title(f"AVG Daily Growth Factor ({area})");
        plt.legend(areas)
        plt.draw()
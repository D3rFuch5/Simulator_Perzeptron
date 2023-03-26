import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mp

plt.style.use('seaborn-v0_8-whitegrid')


class my_Plotting:
    min_x_for_displaying_dataset = 0
    max_x_for_displaying_dataset = 0

    min_y_for_displaying_dataset = 0
    max_y_for_displaying_dataset = 0

    # has to be called last be printed with right dimensions
    @staticmethod
    def plotting_axis_arrows(plt1, ax_annotation_x=r'$x_{1}$', ax_annotation_y=r'$x_{2}$'):

        x_lim = plt1.get_xlim()
        diff_x = abs(x_lim[0] - x_lim[1])

        y_lim = plt1.get_ylim()
        diff_y = abs(y_lim[0] - y_lim[1])

        # Plot x axis
        plt1.arrow(x_lim[0], 0, diff_x, 0,
                   length_includes_head=True,
                   head_width=0.03 * diff_y, head_length=0.03 * diff_x, facecolor='black')

        # Plot y axis
        plt1.arrow(0, y_lim[0], 0,
                   diff_y,
                   length_includes_head=True,
                   head_width=0.03 * diff_x, head_length=0.03 * diff_y, facecolor='black')

        plt1.annotate(ax_annotation_x, xy=(x_lim[1] - 0.02 * diff_x, 0 + 0.02 * diff_y))
        plt1.annotate(ax_annotation_y, xy=(0 + 0.02 * diff_x, y_lim[1] - 0.035 * diff_y))

    @staticmethod
    def calc_dimensions_of_dataset(dataset):
        x = []
        y = []

        for data in dataset:
            x.append(data[0])
            y.append(data[1])

        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)

        diff_x = abs(max_x - min_x)
        diff_y = abs(max_y - min_y)
        # Sets the plotting limits according to the min/max values of the data points
        my_Plotting.min_x_for_displaying_dataset = min_x - np.ceil(0.2 * (diff_x + 1)) - 2
        my_Plotting.max_x_for_displaying_dataset = max_x + np.ceil(0.2 * (diff_x + 1)) + 2

        my_Plotting.min_y_for_displaying_dataset = min_y - np.ceil((0.2 * (diff_y + 1))) - 2
        my_Plotting.max_y_for_displaying_dataset = max_y + np.ceil((0.2 * (diff_y + 1))) + 2

    @staticmethod
    def plotting_data(plt1, headerinfo, dataset, acual_mode):
        # Prepare Plotting
        x = []
        y = []
        colors = []
        for data in dataset:
            x.append(data[0])
            y.append(data[1])
            if acual_mode == "Lineare Klassifikation":
                if data[2] == 0:
                    colors.append("red")
                else:
                    colors.append("green")
            else:
                colors.append("blue")

        # calculated and sets the plotting limits for the data to the calculated limits
        my_Plotting.calc_dimensions_of_dataset(dataset)
        plt1.set_xlim([my_Plotting.min_x_for_displaying_dataset, my_Plotting.max_x_for_displaying_dataset])
        plt1.set_ylim([my_Plotting.min_y_for_displaying_dataset, my_Plotting.max_y_for_displaying_dataset])

        plt1.scatter(x, y, marker='o', c=colors)
        plt1.set_xlabel(headerinfo[0], fontsize=14)
        plt1.set_ylabel(headerinfo[1], fontsize=14)
        plt1.grid(True)

    @staticmethod
    def plotting_separation_line(plt1, weights, threshold, data_loaded=True):
        if data_loaded:
            min_x = my_Plotting.min_x_for_displaying_dataset
            max_x = my_Plotting.max_x_for_displaying_dataset
            min_y = my_Plotting.min_y_for_displaying_dataset
            max_y = my_Plotting.max_y_for_displaying_dataset

        else:
            min_x = -10
            max_x = 10
            min_y = -10
            max_y = 10
        # sets the plotting limits for the line to the calculated limits of the data point
        plt1.set_xlim([min_x, max_x])
        plt1.set_ylim([min_y, max_y])

        x = np.linspace(min_x, max_x)
        y = []
        # if w_y is zero == > handle division by zero
        if weights[1] == 0 and weights[0] == 0:
            x = []
            y = []
        elif weights[1] == 0:
            value = threshold / weights[0]
            x = [value, value]
            # set to y lim
            y = [min_y, max_y]

        else:
            for xCoordinate in x:
                y.append((threshold - (weights[0] * xCoordinate)) / weights[1])

        plt1.plot(x, y, c='blue')

    @staticmethod
    def fill(plt1, weights, threshold, data_loaded=True):
        if data_loaded:
            min_x = my_Plotting.min_x_for_displaying_dataset
            max_x = my_Plotting.max_x_for_displaying_dataset
            min_y = my_Plotting.min_y_for_displaying_dataset
            max_y = my_Plotting.max_y_for_displaying_dataset

        else:
            min_x = -10
            max_x = 10
            min_y = -10
            max_y = 10

        # Falls die Trenngerade parallel zur y-Achse verläuft (bei x = threshold/weights[0]) ...
        # Falls beide Gewichte 0 sind, ist nichts sinnvoll zu plotten
        # Wäre der bedingungslose else - Fall: Hier weggelassen
        if weights[1] == 0 and weights[0] != 0:
            split_at = threshold / weights[0]
            x_1 = np.linspace(min_x, split_at)
            x_2 = np.linspace(split_at, max_x)
            # ... Normalenvektor/Gewichtsvektor zeigt in positive x-Richtung
            if weights[0] > 0:
                plt1.fill_between(x_2, min_y,
                                  max_y, color='green', alpha=0.2,
                                  label="Ausgabe Perzeptron: 1")
                plt1.fill_between(x_1, min_y,
                                  max_y, color='red', alpha=0.2,
                                  label="Ausgabe Perzeptron: 0")
            # ... Normalenvektor/Gewichtsvektor zeigt in negative x-Richtung
            elif weights[0] < 0:
                plt1.fill_between(x_1, min_y,
                                  max_y, color='green', alpha=0.2,
                                  label="Ausgabe Perzeptron: 1")
                plt1.fill_between(x_2, min_y,
                                  max_y, color='red', alpha=0.2,
                                  label="Ausgabe Perzeptron: 0")
            plt1.legend(frameon=True, bbox_to_anchor=(1.12, 1.15), loc="upper right")

        # Falls das Gewicht w_2(weights[1]) positiv ist, zeigt der Normalenvektor auf jeden Fall nach oben
        elif weights[1] > 0:
            x = np.linspace(min_x, max_x)
            y = my_Plotting.separation_function(x, weights, threshold)
            plt1.fill_between(x, y, max_y, color='green', alpha=0.2,
                              label="Ausgabe Perzeptron: 1")
            plt1.fill_between(x, min_y, y, color='red', alpha=0.2,
                              label="Ausgabe Perzeptron: 0")
            plt1.legend(frameon=True, bbox_to_anchor=(1.12, 1.15), loc="upper right")

        # Falls das Gewicht w_2(weights[1]) negativ ist, zeigt der Normalenvektor auf jeden Fall nach unten
        elif weights[1] < 0:
            x = np.linspace(min_x, max_x)
            y = my_Plotting.separation_function(x, weights, threshold)
            plt1.fill_between(x, min_y, y, color='green', alpha=0.2,
                              label="Ausgabe Perzeptron: 1")
            plt1.fill_between(x, y, max_y, color='red', alpha=0.2,
                              label="Ausgabe Perzeptron: 0")
            plt1.legend(frameon=True, bbox_to_anchor=(1.12, 1.15), loc="upper right")

    @staticmethod
    def separation_function(x, weights, threshold):
        if weights[1] == 0:
            y = 0
        else:
            y = (threshold - (weights[0] * x)) / weights[1]
        return y

    @staticmethod
    def emphasize_point(plt1, datapoint):
        plt1.scatter(datapoint[0], datapoint[1], c='yellow', s=200, alpha=0.5, edgecolor='black')

    @staticmethod
    def annotate_next_training_point(plt1, next_data_point):
        plt1.annotate("N\u00E4chster Trainingspunkt", xy=(next_data_point[0], next_data_point[1]),
                      xycoords='data',
                      bbox=dict(boxstyle="round", fc="0.9", ec="black"),
                      xytext=(-10, 40), textcoords='offset points', ha='center',
                      arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))

    @staticmethod
    def plot_point(plt1, datapoint, target):
        if target == 1:
            color = 'green'
        else:
            color = 'red'
        plt1.scatter(datapoint[0], datapoint[1], c='white', alpha=0.9, s=200, edgecolor='blue')
        plt1.scatter(datapoint[0], datapoint[1], c=color, marker='o')

    @staticmethod
    def plotting_gradient_descent_param_updates_lin_classification_with_trace(plot, plot_accuracy, dataset,
                                                                              current_perceptron, old_weights=[]):
        # range_w_1 = np.linspace(act_weights[0] - 7, act_weights[0] + 7, plot_accuracy)
        # range_w_2 = np.linspace(act_weights[1] - 7, act_weights[1] + 7, plot_accuracy)

        range_w_1 = np.linspace(-10, 10, plot_accuracy)
        range_w_2 = np.linspace(-10, 10, plot_accuracy)

        threshold = current_perceptron.threshold

        input_x_1 = dataset[0]
        input_x_2 = dataset[1]
        target = dataset[2]
        Z = []
        for i in range(len(range_w_1)):
            z = []
            for j in range(len(range_w_2)):
                if (range_w_1[j] * input_x_1 + range_w_2[i] * input_x_2) >= threshold:
                    output = 1
                else:
                    output = 0
                temp = target - output

                z.append(temp)

            Z.append(z)

        contourspaces = plot.contourf(range_w_1, range_w_2, Z, levels=[-2, -1, 0, 1],
                                      colors=['yellow', 'grey', 'yellow'],
                                      alpha=0.4)
        plot.set_xlabel("$w_{1}$", fontsize=14)
        plot.set_ylabel("$w_{2}$", fontsize=14)
        artists, dummy = contourspaces.legend_elements()
        plot.legend(artists[0:len(artists) - 1], ['Falsch klassifiziert', 'Richtig klassifiziert'], frameon=True,
                    bbox_to_anchor=(1.12, 1.15), loc="upper right")
        plot.grid(True)

        current_weights_history = current_perceptron.weights_history
        # trace of old weights
        for data in current_weights_history:
            plot.scatter(data[0], data[1], c='blue', alpha=0.3)

        if len(current_weights_history) > 1:
            for i in range(1, len(current_weights_history)):
                plot.arrow(current_weights_history[i - 1][0], current_weights_history[i - 1][1],
                           current_weights_history[i][0] - current_weights_history[i - 1][0],
                           current_weights_history[i][1] - current_weights_history[i - 1][1], linewidth=1,
                           head_width=0.2, head_length=0.3,
                           head_starts_at_zero=False, length_includes_head=True, color="blue", alpha=0.7)

        # point for actual weight and actual bias
        plot.scatter(current_perceptron.weights[0], current_perceptron.weights[1], c='blue', s=20)

        if old_weights != []:
            if not (current_perceptron.weights[0] == old_weights[0] and current_perceptron.weights[1] == old_weights[
                1]):
                # point for old weight and bias and vector
                diff_x = current_perceptron.weights[0] - old_weights[0]
                diff_y = current_perceptron.weights[1] - old_weights[1]
                plot.arrow(old_weights[0], old_weights[1], diff_x, diff_y, linewidth=2, head_width=0.2, head_length=0.3,
                           head_starts_at_zero=False, length_includes_head=True, color="black")

                plot.arrow(old_weights[0], old_weights[1], diff_x, 0, linewidth=1, head_width=0.2, head_length=0.3,
                           head_starts_at_zero=False, length_includes_head=True, color="black",
                           linestyle='dotted', alpha=0.7)
                plot.arrow(old_weights[0] + diff_x, old_weights[1], 0, diff_y, linewidth=1, head_width=0.2,
                           head_length=0.3,
                           head_starts_at_zero=False, length_includes_head=True, color="black",
                           linestyle='dotted', alpha=0.7)

                plot.annotate(str(diff_x),
                              xy=(old_weights[0] + (diff_x / 2), old_weights[1] + 0.2))
                plot.annotate(str(diff_y),
                              xy=(old_weights[0] + diff_x + 0.2, old_weights[1] + (diff_y / 2)))

    @staticmethod
    def enable_interactive_plot(enable):
        if not enable:
            plt.ioff()
        else:
            plt.ion()

    @staticmethod
    def plotting_gradient_descent_param_updates_lin_classification_with_trace_3D(plot, plot_accuracy, datapoint,
                                                                                 current_perceptron):
        plot_range_min = -10
        plot_range_max = 10

        # Bereiche festlegen, in denen die Gewichte/Schwellenwert laufen sollen
        range_w_1 = np.linspace(plot_range_min, plot_range_max, plot_accuracy)
        range_w_2 = np.linspace(plot_range_min, plot_range_max, plot_accuracy)
        range_s = np.linspace(plot_range_min, plot_range_max, plot_accuracy)

        # Erstellen der Kombinationsgitter
        possible_w1, possible_w2, possible_s = np.meshgrid(range_w_1, range_w_2, range_s, indexing='ij')

        # Berechnung der zusammengehörigen Koordinaten.
        # Benötigt für die Bestimmung des konkreten Fehlers(d.h. hier, ob korrekt klassifiziert oder nicht)
        coordinates = []
        for a, b, c in zip(possible_w1, possible_w2, possible_s):
            for a1, b1, c1 in zip(a, b, c):
                for a2, b2, c2 in zip(a1, b1, c1):
                    coordinates.append([a2, b2, c2])

        # Speichert den jeweiligen Fehler der Gewichts/Schwellenwertkombinationen aus coordinates
        error_for_coordinate_list = []
        target = datapoint[-1]
        input_data = datapoint[:-1]
        for parameters in coordinates:
            weighted_sum = 0
            for wi, i in zip(parameters[:-1], input_data):
                weighted_sum += wi * i

            # Aktivierungsfunktion
            if weighted_sum >= parameters[-1]:
                calculated_output = 1
            else:
                calculated_output = 0

            # Vergleich, ob ein Fehler vorliegt und passend hinzufügen
            if target != calculated_output:
                error_for_coordinate_list.append(1)
            else:
                error_for_coordinate_list.append(0)

        # Festlegungen für die grafische Darstellung
        colormap = plt.cm.RdYlGn_r
        norm = plt.Normalize(vmin=0, vmax=max(error_for_coordinate_list))
        alpha_value = 0.2
        plot.set_xlabel("$w_{1}$", fontsize=14)
        plot.set_ylabel("$w_{2}$", fontsize=14)
        plot.set_zlabel("\u03B8", fontsize=14)

        # Lokales Speichern der Gewichts-Historie damit nicht immer abgefragt werden muss
        current_weights_history = current_perceptron.weights_history

        # Plotten der bisherigen Gewichts/Schwellenwertkombinationen als Punkte
        for parameters in current_weights_history[:-1]:
            plot.scatter(parameters[0], parameters[1], parameters[2], c='blue', alpha=0.5)

        # Plotten der aktuellen Gewichts/Schwellenwertkombination als Punkt
        plot.scatter(current_perceptron.weights[0], current_perceptron.weights[1], current_perceptron.threshold,
                     c='blue', s=20)

        # Plotten der Vektoren zwischen den Punkten
        if len(current_weights_history) > 1:
            for i in range(0, len(current_weights_history) - 1):
                plot.quiver(current_weights_history[i][0], current_weights_history[i][1], current_weights_history[i][2],
                            current_weights_history[i + 1][0] - current_weights_history[i][0],
                            current_weights_history[i + 1][1] - current_weights_history[i][1],
                            current_weights_history[i + 1][2] - current_weights_history[i][2], color='blue')

        # Plotten der Punkte aller möglichen Gewichts/Schwellenwertkombinationen als Punkte
        # Die Farbe hängt vom berechneten Fehler ab
        plot.scatter(possible_w1, possible_w2, possible_s, c=error_for_coordinate_list, cmap=colormap, norm=norm, s=25,
                     edgecolor='none', alpha=alpha_value)

        # Plotten der Trennebene
        N = 101
        range_x = np.linspace(plot_range_min, plot_range_max, N)
        range_y = np.linspace(plot_range_min, plot_range_max, N)

        possible_x, possible_y = np.meshgrid(range_x, range_y)
        calculated_z_values = possible_x * input_data[0] + possible_y * input_data[1]
        calculated_z_values[calculated_z_values < plot_range_min] = np.nan
        calculated_z_values[calculated_z_values > plot_range_max] = np.nan
        plot.plot_surface(possible_x, possible_y, calculated_z_values, color="black", alpha=0.3)

        # x, y, z = possible_x.flatten(), possible_y.flatten(), calculated_z_values.flatten()
        # usable_points = (plot_range_min < z) & (z < plot_range_max)
        # x, y, z = x[usable_points], y[usable_points], z[usable_points]
        # plot.plot_trisurf(x, y, z, color="black", alpha=0.3)

        # Plotten der Legende
        red_patch = mp.Patch(color='#c72c52', label='Fehlerhafte Klassifikation')
        green_patch = mp.Patch(color='#329c69', label='Korrekte Klassifikation')
        simArtist = plt.Line2D((0, 0), (0, 1), label='Gewichte/Schwellenwert', color='blue', marker='o', linestyle='')
        plot.legend(handles=[red_patch, green_patch, simArtist], frameon=True, bbox_to_anchor=(0.68, 1.15),
                    loc="upper left")

    # Regression: Hier wird mit dem BIAS gearbeitet
    @staticmethod
    def plotting_regression_line(current_plot, weights, bias):

        min_x = my_Plotting.min_x_for_displaying_dataset
        max_x = my_Plotting.max_x_for_displaying_dataset
        min_y = my_Plotting.min_y_for_displaying_dataset
        max_y = my_Plotting.max_y_for_displaying_dataset

        # sets the plotting limits for the line to the calculated limits of the data point
        current_plot.set_xlim([min_x, max_x])
        current_plot.set_ylim([min_y, max_y])

        x = np.linspace(min_x, max_x)
        y = []
        # if w_y is zero == > handle division by zero
        if weights[0] == 0 and bias == 0:
            x = []
            y = []
        else:
            for xCoordinate in x:
                y.append((weights[0] * xCoordinate + bias))

        current_plot.plot(x, y, c='blue')

    # Berechnet den quad. Gesamtfehler für den gesamten Datensatz und die übergebenen Gewichte
    @staticmethod
    def calc_loss_for_dataset(dataset, current_w1, current_w0):
        error = 0.0
        for data in dataset:
            input_data = data[0:(len(data) - 1)]
            target = data[len(data) - 1]
            output = current_w1 * input_data[0] + current_w0
            error += ((target - output) ** 2)
        return error

    # Regression: Hier wird mit dem BIAS gearbeitet
    @staticmethod
    def plotting_loss_surface(current_plot, current_perceptron, training_data):
        range_w0 = np.linspace(-25, 25, 25)
        range_w1 = np.linspace(-25, 25, 25)
        possible_w0, possible_w1 = np.meshgrid(range_w0, range_w1, indexing='xy')

        # Berechnet den quad. Gesamtfehler für jede mögliche Kombination an Gewichten
        grid = []
        for current_w1 in range_w1:
            subgrid = []
            for current_w0 in range_w0:
                subgrid.append(my_Plotting.calc_loss_for_dataset(training_data, current_w1, current_w0))
            grid.append(subgrid)

        errors = np.array(grid)

        # Hier ist der threshold eigentlich ein Bias
        current_plot.scatter(current_perceptron.threshold, current_perceptron.weights[0],
                             my_Plotting.calc_loss_for_dataset(training_data,
                                                               current_perceptron.weights[0],
                                                               current_perceptron.threshold),
                             c='red', s=10)

        current_plot.plot_wireframe(possible_w0, possible_w1, errors, color='blue', alpha=0.2)
        current_plot.set_xlabel("$w_{0}$", fontsize=14)
        current_plot.set_ylabel("$w_{1}$", fontsize=14)
        current_plot.legend(["Verlust für aktuelle Gewichte", "Verlustfunktion"], frameon=True,
                            bbox_to_anchor=(0.68, 1.15),
                            loc='upper left')

    @staticmethod
    def plotting_weights_list(plot_gradient_descent, weights_list):
        # Prepare Plotting
        x = []
        y = []
        colors = []
        for data in weights_list:
            x.append(data[0])
            y.append(data[1])

        # sets the plotting limits for the data to the calculated limits
        plot_gradient_descent.set_xlim([-10, 10])
        plot_gradient_descent.set_ylim([-10, 10])
        plot_gradient_descent.scatter(x, y, marker='o')
        plot_gradient_descent.grid(True)

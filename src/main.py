import tkinter
from tkinter.filedialog import askopenfilename
from tkinter import messagebox

from src.my_Plotting import my_Plotting
from src.perceptron import perceptron
from src.gui_Simulator import gui_Simulator

import src.csv_Reader as csv_Reader


class main:
    # Default-Werte für wichtige Parameter
    default_WEIGHT_1 = 1
    default_WEIGHT_2 = 1
    default_THRESHOLD = 1
    default_LEARNING_RATE_LINEAR_REGRESSION = 0.001
    default_LEARNING_RATE_LINEAR_CLASSIFICATION = 1
    default_EPOCHS = 1

    default_CLASSIFICATION_POINT_INPUT_X1 = 1
    default_CLASSIFICATION_POINT_INPUT_X2 = 1
    default_CLASSIFICATION_POINT_TARGET = 1

    mode_LINEAR_CLASSIFICATION = "Lineare Klassifikation"
    mode_LINEAR_REGRESSION = "Lineare Regression"

    def __init__(self):
        self.gui = gui_Simulator(self)

        self.initialized_weight_1 = self.default_WEIGHT_1
        self.initialized_weight_2 = self.default_WEIGHT_2
        self.initialized_threshold = self.default_THRESHOLD
        if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            self.initialized_learning_rate = self.default_LEARNING_RATE_LINEAR_CLASSIFICATION
        else:
            self.initialized_learning_rate = self.default_LEARNING_RATE_LINEAR_REGRESSION

        self.selected_filepath = self.gui.entered_filepath

        self.current_perceptron = None
        self.initialized_with_learning_rate = False
        self.read_in_train_data_header = None
        self.read_in_train_data = None

        self.index_of_dataset_to_train = 0
        self.size_training_data = 0

        self.data_loaded = False

        self.learning_rate_info_already_shown = False

        # phase_0 - Anzeige von "Neue Klassifikation starten"/Klassifikationsergebnis(nach einem Durchlauf)
        # phase_1 - Einlesen des Punkts und Button "Punkt anzeigen"
        # phase_2 - Anzeigen des Punkts, Button "Berechnungen"
        # phase_3 - Anzeigen der Berechnungen, Button Vergleich
        self.classification_phase = "phase_0"
        self.point_to_classify_x_1 = self.default_CLASSIFICATION_POINT_INPUT_X1
        self.point_to_classify_x_2 = self.default_CLASSIFICATION_POINT_INPUT_X2
        self.point_to_classify_target = self.default_CLASSIFICATION_POINT_TARGET

        # Da alle Eingabemöglichkeiten während des Automodes gesperrt werden, kann dieser nur in
        # call_train_automode gesetzt werden
        self.auto_training_running = False
        self.auto_mode_running_speed = 1000
        self.callID = None

    def open_file(self, lbl):
        """
        Liest die Daten aus dem gewählten CSV-File aus, belegt die entsprechenden Attribute und
        plottet die Daten. Setzt außerdem die Anzeige auf den Initialzustand des aktuell gewählten Modus zurück
        Zeigt den Pfad der ausgewählten Datei in der Oberfläche an
        Prüft außerdem, ob die gelesenen Daten Zahlen sind
        :param lbl: Label, in dem der Pfad angezeigt werden soll
        """

        self.read_in_train_data = None
        self.read_in_train_data_header = None
        self.data_loaded = False
        self.selected_filepath = ""
        self.index_of_dataset_to_train = 0
        self.size_training_data = 0

        # Zurücksetzen der Daten des bisherigen Lernprozesses des Perzeptrons
        if self.current_perceptron is not None:
            self.current_perceptron.number_of_training_steps = 0
            self.current_perceptron.weights_history = [[w for w in self.current_perceptron.weights]]
            self.current_perceptron.weights_history[0].append(self.current_perceptron.threshold)
            self.current_perceptron.error = 0

        if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:

            self.gui.adapt_perceptron_control_buttons(self.mode_LINEAR_CLASSIFICATION,
                                                      self.gui.classification_point_enabled.get())

            self.gui.current_training_steps.set(0)

            # Reinigt die aktuelle Plotting-Figure und setzt die Darstellung auf den 2D-Plot
            self.gui.reset_plotting_figure("2D")
            self.gui.canvas.draw()

            callback_func = self.gui.selected_display_mode_for_linear_classification.trace_info()[0]
            self.gui.selected_display_mode_for_linear_classification.trace_remove(callback_func[0],
                                                                                  callback_func[1])
            self.gui.selected_display_mode_for_linear_classification.set("Trenngerade")
            self.gui.selected_display_mode_for_linear_classification.trace_add(mode="write",
                                                                               callback=self.call_display_mode_switched_linear_classification)

            self.gui.classification_point_enabled.set(0)
            self.gui.adapt_classification_of_a_point_control(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                             display_mode="Trenngerade")

            self.gui.adapt_plot_contol_options(self.mode_LINEAR_CLASSIFICATION,
                                               self.gui.classification_point_enabled.get(),
                                               self.gui.selected_display_mode_for_linear_classification.get())

            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)
            self.build_training_protocol(clear_explanation=True)


        elif self.gui.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            # Es wird das Perzeptron zurückgesetzt. Neue Daten brauchen neues Training -> Designentscheidung
            # Werte des Perzeptrons bleiben. perzeptron_initialized kann nur durch Neuinitialisierung auf true
            # gesetzt werden
            self.current_perceptron = None
            self.initialized_with_learning_rate = False

            self.gui.adapt_perceptron_control_buttons(self.mode_LINEAR_REGRESSION)

            self.gui.current_training_steps.set(0)

            # Reinigt die aktuelle Plotting-Figure und setzt die Darstellung auf den 2D-Plot
            self.gui.reset_plotting_figure("2D")
            self.gui.canvas.draw()

            callback_func = self.gui.selected_display_mode_for_linear_regression.trace_info()[0]
            self.gui.selected_display_mode_for_linear_regression.trace_remove(callback_func[0], callback_func[1])
            self.gui.selected_display_mode_for_linear_regression.set("Geradenansicht")
            self.gui.selected_display_mode_for_linear_regression.trace_add(mode='write',
                                                                           callback=self.call_display_mode_switched_linear_regression)

            self.gui.adapt_plot_contol_options(self.mode_LINEAR_REGRESSION)

            # Zurücksetzen des Perzeptronsanzeige
            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)
            self.build_explanation(current_mode=self.mode_LINEAR_REGRESSION, clear_explanation=True)

        try:
            # Versuch die Datei zu öffnen und zu lesen
            filepath = askopenfilename(
                filetypes=[("CSV Files", "*.csv")])
            self.selected_filepath = filepath
            self.read_in_train_data_header = csv_Reader.read_in_csv_header(self.selected_filepath)
            self.read_in_train_data = csv_Reader.read_in_csv(self.selected_filepath)
            self.size_training_data = len(self.read_in_train_data)
            self.data_loaded = True
            lbl["text"] = f"{self.selected_filepath}"

            if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
                if self.current_perceptron is not None:
                    self.gui.replot(header_data=self.read_in_train_data_header,
                                    training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)
                    self.gui.update_progressbar(
                        self.current_perceptron.number_of_correct_classified(self.read_in_train_data),
                        self.size_training_data)
                else:
                    self.gui.update_progressbar(0, 1)

                    my_Plotting.plotting_data(self.gui.plot2D, self.read_in_train_data_header,
                                              self.read_in_train_data, self.gui.selected_mode.get())
                    my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                    self.gui.canvas.draw()
            elif self.gui.selected_mode.get() == self.mode_LINEAR_REGRESSION:
                my_Plotting.plotting_data(self.gui.plot2D, self.read_in_train_data_header,
                                          self.read_in_train_data, self.gui.selected_mode.get())
                my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                self.gui.canvas.draw()

        except Exception as e:
            self.data_loaded = False
            self.selected_filepath = ""
            lbl["text"] = f"{self.selected_filepath}"
            err_msg = "Es ist ein Fehler aufgetreten!"
            if type(e) == ValueError:
                err_msg = "Der gewählte Datensatz enthält Einträge, die keine Zahlen darstellen."
            if type(e) == FileNotFoundError:
                err_msg = "Es wurde keine Datei ausgewählt!"
            messagebox.showerror(title="Fehler", message=err_msg, parent=self.gui.window)

            self.gui.show_next_point_to_train.set(0)
            # Im Modus "Lineare Klassifikation" kann das Perzeptron auch ohne Trainingsdaten initialisiert werden
            # Damit das Perzeptron bei einem Fehler auch wieder geplottet wird, dieser Fall
            if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION and self.current_perceptron is not None:
                my_Plotting.plotting_separation_line(self.gui.plot2D, self.current_perceptron.weights,
                                                     self.current_perceptron.threshold, data_loaded=False)
                my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                self.gui.canvas.draw()

    def call_general_mode_switch(self, dummy1, dummy2, dummy3):
        """
        Setzt die Anzeige auf den Initialzustand des aktuell ausgewählten Modus zurück
        Des Weiteren werden alle inneren Parameter und das Perzeptron zurückgesetzt
        """
        self.read_in_train_data = None
        self.read_in_train_data_header = None
        self.data_loaded = False
        self.current_perceptron = None
        self.initialized_with_learning_rate = False

        self.gui.use_learning_rate.set(0)
        self.gui.classification_point_enabled.set(0)

        # resets perceptron with entered values
        if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            # sets the default values for the entry fields
            self.gui.entered_w_1.set(self.default_WEIGHT_1)
            self.gui.entered_w_2.set(self.default_WEIGHT_2)
            self.gui.entered_threshold.set(self.default_THRESHOLD)
            self.gui.entered_learning_rate.set(self.default_LEARNING_RATE_LINEAR_CLASSIFICATION)

            # resets the initialized params to defaults
            self.initialized_weight_1 = self.default_WEIGHT_1
            self.initialized_weight_2 = self.default_WEIGHT_2
            self.initialized_threshold = self.default_THRESHOLD
            self.initialized_learning_rate = self.default_LEARNING_RATE_LINEAR_CLASSIFICATION

            callback_func = self.gui.selected_display_mode_for_linear_classification.trace_info()[0]
            self.gui.selected_display_mode_for_linear_classification.trace_remove(callback_func[0], callback_func[1])
            self.gui.selected_display_mode_for_linear_classification.set("Trenngerade")
            self.gui.selected_display_mode_for_linear_classification.trace_add(mode="write",
                                                                               callback=self.call_display_mode_switched_linear_classification)
            self.gui.update_progressbar(0, 1)
            self.gui.current_training_steps.set(0)

            self.gui.show_half_area.set(0)
            self.gui.show_next_point_to_train.set(0)

            self.index_of_dataset_to_train = 0
            self.size_training_data = 0
            # Zurücksetzen der Perzeptronanzeoge
            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)
            self.build_training_protocol(clear_explanation=True)

        elif self.gui.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            # sets the default values for the entry fields
            self.gui.entered_w_1.set(self.default_WEIGHT_1)
            self.gui.entered_threshold.set(self.default_THRESHOLD)
            self.gui.entered_learning_rate.set(self.default_LEARNING_RATE_LINEAR_REGRESSION)

            # resets the initialized params to defaults
            self.initialized_weight_1 = self.default_WEIGHT_1
            self.initialized_threshold = self.default_THRESHOLD
            self.initialized_learning_rate = self.default_LEARNING_RATE_LINEAR_REGRESSION

            callback_func = self.gui.selected_display_mode_for_linear_regression.trace_info()[0]
            self.gui.selected_display_mode_for_linear_regression.trace_remove(callback_func[0], callback_func[1])
            self.gui.selected_display_mode_for_linear_regression.set("Geradenansicht")
            self.gui.selected_display_mode_for_linear_regression.trace_add(mode='write',
                                                                           callback=self.call_display_mode_switched_linear_regression)
            self.gui.current_training_steps.set(0)

            # Only for reseting display of perceptron
            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)

        # adapts gui according to the seleceted mode
        self.gui.adapt_gui_for_mode()

        # Reinigt die akutelle Plotting-Figure und setzt die Darstellung auf den 2D-Plot
        self.gui.reset_plotting_figure("2D")
        self.gui.canvas.draw()

        self.build_explanation(clear_explanation=True, current_mode=self.gui.selected_mode.get())

    def call_display_mode_switched_linear_classification(self, dummy1, dummy2, dummy3):
        """
        Regelt die Anzeige des Anzeigemodus des Plots im Modus "Lineare Klassifikation".
        In Fehlerfällen wird eine Fehlermeldung ausgegeben und die Anzeigeauswahl auf den 2D-Fall des
        jeweiligen Modus gesetzt. (Replot, ... finden nicht statt. Man geht davon aus, dass die Anzeige in einem
        konsistenten Zustand ist)
        """
        if self.current_perceptron is not None:
            current_display_mode = self.gui.selected_display_mode_for_linear_classification.get()
            if self.data_loaded:
                self.gui.adapt_perceptron_control_buttons(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                          classification_of_a_point_enabled=self.gui.classification_point_enabled.get(),
                                                          display_mode=self.gui.selected_display_mode_for_linear_classification.get())

                self.gui.adapt_classification_of_a_point_control(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                                 display_mode=current_display_mode,
                                                                 classification_of_a_point_enabled=self.gui.classification_point_enabled.get())
                self.gui.adapt_plot_contol_options(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                   classification_of_a_point_enabled=self.gui.classification_point_enabled.get(),
                                                   display_mode=current_display_mode)
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                                emphasize_point=False,
                                data_point=self.read_in_train_data[self.index_of_dataset_to_train])

            else:
                if current_display_mode == "Gewichtsaktualisierung":
                    messagebox.showerror(title="Fehler", message="Keine Daten geladen!", parent=self.gui.window)

                    callback_func = self.gui.selected_display_mode_for_linear_classification.trace_info()[0]
                    self.gui.selected_display_mode_for_linear_classification.trace_remove(callback_func[0],
                                                                                          callback_func[1])
                    self.gui.selected_display_mode_for_linear_classification.set("Trenngerade")
                    self.gui.selected_display_mode_for_linear_classification.trace_add(mode="write",
                                                                                       callback=self.call_display_mode_switched_linear_classification)
        else:
            messagebox.showerror(title="Fehler", message="Das Perzeptron ist nicht initialisiert!",
                                 parent=self.gui.window)

            callback_func = self.gui.selected_display_mode_for_linear_classification.trace_info()[0]
            self.gui.selected_display_mode_for_linear_classification.trace_remove(callback_func[0],
                                                                                  callback_func[1])
            self.gui.selected_display_mode_for_linear_classification.set("Trenngerade")
            self.gui.selected_display_mode_for_linear_classification.trace_add(mode="write",
                                                                               callback=self.call_display_mode_switched_linear_classification)

    def call_display_mode_switched_linear_regression(self, dummy1, dummy2, dummy3):
        """
        Regelt die Anzeige des Anzeigemodus des Plots im Modus "Lineare Klassifikation".
        In Fehlerfällen wird eine Fehlermeldung ausgegeben und die Anzeigeauswahl auf den 2D-Fall des
        jeweiligen Modus gesetzt. (Replot, ... finden nicht statt. Man geht davon aus, dass die Anzeige in einem
        konsistenten Zustand ist)
        """
        if self.current_perceptron is not None and self.data_loaded:
            self.gui.replot(header_data=self.read_in_train_data_header,
                            training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)

        else:
            if self.gui.selected_display_mode_for_linear_regression.get() == "Gradientenabstieg":

                error_msg = ""
                if not self.data_loaded:
                    error_msg += "Keine Trainingsdaten geladen.\n"
                if self.current_perceptron is None:
                    error_msg += "Das Perzeptron ist nicht initialisiert."
                messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

                callback_func = self.gui.selected_display_mode_for_linear_regression.trace_info()[0]
                self.gui.selected_display_mode_for_linear_regression.trace_remove(callback_func[0],
                                                                                  callback_func[1])
                self.gui.selected_display_mode_for_linear_regression.set("Geradenansicht")
                self.gui.selected_display_mode_for_linear_regression.trace_add(mode="write",
                                                                               callback=self.call_display_mode_switched_linear_regression)

    # NICHT VERWENDET
    def call_display_mode_switched(self, dummy1, dummy2, dummy3):
        """
        Regelt die Anzeige des Anzeigemodus des Plots im jeweils ausgewählten Modus(Lineare Klassifiaktion, Lineare Regression)
        Sind Daten geladen und das Perzeptron initialisiert, wird die Anzeige entsprechend des ausgewählten Modus
        angepasst. In allen anderen Fällen wird eine Fehlermeldung ausgegeben und die Anzeigeauswahl auf den 2D-Fall des
        jeweiligen Modus gesetzt. (Replot, ... finden nicht statt. Man geht davon aus, dass die Anzeige in einem
        konsistenten Zustand ist)
        """
        current_mode = self.gui.selected_mode.get()

        if self.data_loaded and self.current_perceptron is not None:
            if current_mode == self.mode_LINEAR_CLASSIFICATION:

                current_display_mode = self.gui.selected_display_mode_for_linear_classification.get()
                self.gui.adapt_classification_of_a_point_control(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                                 display_mode=current_display_mode,
                                                                 classification_of_a_point_enabled=self.gui.classification_point_enabled.get())
                self.gui.adapt_plot_contol_options(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                   classification_of_a_point_enabled=self.gui.classification_point_enabled.get(),
                                                   display_mode=current_display_mode)
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                                emphasize_point=False,
                                data_point=self.read_in_train_data[self.index_of_dataset_to_train])
            # Da im Modus "Lineare Regression" die alle Anzeigemodi die identische Oberfläche haben, muss nur der
            # Plot aktualisiert werden. Die Anzeige der korrekten Oberfläche muss beim Wechsel des Modus hergestellt werden
            elif current_mode == self.mode_LINEAR_REGRESSION:
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)

        # Falls das Perzeptron nicht initialisiert oder keine Daten geladen sind, wird eine Fehlermeldung angezeigt
        # und die Auswahl zurückgesetzt
        else:
            error_msg = ""
            if not self.data_loaded:
                error_msg += "Keine Trainingsdaten geladen.\n"
            if self.current_perceptron is None:
                error_msg += "Das Perzeptron ist nicht initialisiert."
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

            if current_mode == self.mode_LINEAR_CLASSIFICATION:
                callback_func = self.gui.selected_display_mode_for_linear_classification.trace_info()[0]
                self.gui.selected_display_mode_for_linear_classification.trace_remove(callback_func[0],
                                                                                      callback_func[1])
                self.gui.selected_display_mode_for_linear_classification.set("Trenngerade")
                self.gui.selected_display_mode_for_linear_classification.trace_add(mode="write",
                                                                                   callback=self.call_display_mode_switched)
                self.gui.com_box_lin_classification_display_mode['takefocus'] = False
            elif current_mode == self.mode_LINEAR_REGRESSION:
                callback_func = self.gui.selected_display_mode_for_linear_regression.trace_info()[0]
                self.gui.selected_display_mode_for_linear_regression.trace_remove(callback_func[0],
                                                                                  callback_func[1])
                self.gui.selected_display_mode_for_linear_regression.set("Geradenansicht")
                self.gui.selected_display_mode_for_linear_regression.trace_add(mode="write",
                                                                               callback=self.call_display_mode_switched)
                self.gui.com_box_lin_regression_display_mode['takefocus'] = False

    def call_replot_show_half_areas(self):
        """
        Regelt die Anzeige der Halbräume im Plot.
        Methode wird nur im Modus "Lineare Klassifikation" mit Anzeigemodus "Trenngerade" verwendet.
        """
        if self.current_perceptron is not None:
            if self.data_loaded:
                if self.gui.classification_point_enabled.get() == 1:
                    self.gui.replot(header_data=self.read_in_train_data_header,
                                    training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                                    point_to_classify=[[self.point_to_classify_x_1, self.point_to_classify_x_2],
                                                       self.point_to_classify_target])
                else:
                    self.gui.replot(header_data=self.read_in_train_data_header,
                                    training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)

            # Keine Daten geladen
            else:
                # Clear, damit nicht mehrfach die Halbräume übereinander gezeichnet werden bei ein- und ausschalten,
                # wenn nur die Trenngerade gezeichnet ist oder klassifiziert wird
                self.gui.clear_plot()

                # Zeichnen des Plots
                if self.gui.show_half_area.get() == 1:
                    my_Plotting.fill(self.gui.plot2D, self.current_perceptron.weights,
                                     self.current_perceptron.threshold,
                                     data_loaded=False)
                my_Plotting.plotting_separation_line(self.gui.plot2D, self.current_perceptron.weights,
                                                     self.current_perceptron.threshold, data_loaded=False)
                my_Plotting.plotting_axis_arrows(self.gui.plot2D)

                # Falls Klassifikation aktiviert ist, muss auch der zu klassifizierende Punkt geplottet werden
                if self.gui.classification_point_enabled.get() == 1 and (not self.classification_phase == "phase_1"):
                    my_Plotting.plot_point(self.gui.plot2D, [self.point_to_classify_x_1, self.point_to_classify_x_2],
                                           self.point_to_classify_target)
                self.gui.canvas.draw()
        # Falls kein Perzeptron initialisiert ist, passiert nichts. Es wird davon ausgegangen, dass sich die Oberfläche
        # in einem konsistenen Zustand befindet
        else:
            self.gui.show_half_area.set(0)
            messagebox.showerror(title="Fehler", message="Das Perzeptron ist nicht initialisiert!",
                                 parent=self.gui.window)

    def call_replot_show_next_point_to_train(self):
        """
        Regelt die Anzeige der Annotation "Näcchster Trainingsdatenpunkt" im Plot.
        Methode wird nur im Modus "Lineare Klassifikation" mit Anzeigemodus "Trenngerade" verwendet
        """
        if self.current_perceptron is not None and self.data_loaded:
            # Klassifikation aktiviert
            if self.gui.classification_point_enabled.get() == 1:
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                                point_to_classify=[[self.point_to_classify_x_1, self.point_to_classify_x_2],
                                                   self.point_to_classify_target])
            # Klassifikation nicht aktiviert
            else:
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)
        else:
            self.gui.show_next_point_to_train.set(0)
            error_msg = ""
            if not self.data_loaded:
                error_msg += "Keine Trainingsdaten geladen.\n"
            if self.current_perceptron is None:
                error_msg += "Das Perzeptron ist nicht initialisiert."
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    # NICHT VERWENDET
    def call_replot_for_half_areas_next_point_to_train(self):
        if self.data_loaded and self.current_perceptron is not None:
            # Klassifikation aktiviert
            if self.gui.classification_point_enabled.get() == 1:
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                                point_to_classify=[[self.point_to_classify_x_1, self.point_to_classify_x_2],
                                                   self.point_to_classify_target])
            # Klassifikation nicht aktiviert
            else:
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)

            # Guarantees that the Train All button stays is enabled or disabled
            self.gui.adapt_perceptron_control_buttons(self.mode_LINEAR_CLASSIFICATION,
                                                      self.gui.classification_point_enabled.get())
        # Wenn nur die Trenngerade gezeichnet werden soll, weil noch keine Daten geladen sind
        # Hier muss manuell gearbeitet werden. Standardfall(initialisiert und Daten geladen) erledigt replot
        elif self.current_perceptron is not None and not self.data_loaded:
            # Ausführung nur, wenn der nächste Datenpunkt nicht angezeigt werden soll
            # Da keine Daten geladen wurden, kann dieser nicht angezeigt werden
            if self.gui.show_next_point_to_train.get() == 0:
                # Clear, damit nicht mehrfach die Halbräume übereinander gezeichnet werden bei ein und ausschalten, wenn nur
                # die Trenngerade gezeichnet ist oder klassifiziert wird
                self.gui.clear_plot()
                if self.gui.show_half_area.get() == 1:
                    my_Plotting.fill(self.gui.plot2D, self.current_perceptron.weights,
                                     self.current_perceptron.threshold,
                                     data_loaded=False)
                my_Plotting.plotting_separation_line(self.gui.plot2D, self.current_perceptron.weights,
                                                     self.current_perceptron.threshold, data_loaded=False)
                if self.gui.classification_point_enabled.get() == 1 and (
                        not self.classification_phase == "phase_0"):
                    my_Plotting.plot_point(self.gui.plot2D, [self.point_to_classify_x_1, self.point_to_classify_x_2],
                                           self.point_to_classify_target)
                my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                self.gui.canvas.draw()
            # Behandlung von Anzeige des nächsten Datenpunkts, wenn keine Daten geladen wurden
            else:
                self.gui.show_next_point_to_train.set(0)
                error_msg = ""
                if not self.data_loaded:
                    error_msg += "Keine Trainingsdaten geladen."
                messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)
        else:
            self.gui.show_half_area.set(0)
            self.gui.show_next_point_to_train.set(0)
            error_msg = ""
            if not self.data_loaded:
                error_msg += "Keine Trainingsdaten geladen.\n"
            if self.current_perceptron is None:
                error_msg += "Das Perzeptron ist nicht initialisiert."
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    def call_switch_learning_rate_usage(self):
        """
        Regelt die Anzeige der Lernrateneingabe
        Wenn die Lernrate eingeschaltet wird, wird die Oberfläche angepasst um die Lernrate einzugeben
        Die eingegebene Lernrate wird erst nach Neuinitialisierung des Perzeptrons aktiv
        """
        if not self.learning_rate_info_already_shown:
            messagebox.showinfo(title="Verwendung der Lernrate", message="Eine Änderung der Lernrate wird erst bei "
                                                                         "(Neu-) Initialisierung des Perzeptrons "
                                                                         "wirksam.", parent=self.gui.window)
            self.learning_rate_info_already_shown = True

        # Einlesen und überprüfen der Eingabe wird erst beim Initialisieren in beim Prüfen der Eingaben durchgeführt
        # Es wird nur ein Aufruf der GUI-Anpassung durchgeführt
        self.gui.adapt_gui_for_learning_rate()

    def check_inputs(self, selected_mode):
        """
        Liest die eingegebenen Werte für w_1, w_2,s (bzw. w_1, w_0) und ggf. die Lernrate ein, prüft die
        Eingaben auf Zulässigkeit und setzt die Werte im Programm. Tritt ein Fehler auf, wird der jeweilige
        Wert auf den Default-Wert gesetzt und die zurückzugebende Fehlermeldung ergänzt.
        :param selected_mode: Momentan ausgewählter Modus (Lineare Klassifikation, Lineare Regression)
        :return: Text, welcher die Fehlermeldungen enthält
        """
        error_msg = ""
        try:
            self.initialized_weight_1 = int(self.gui.entered_w_1.get())
        except:
            error_msg += "Das eingegebene Gewicht w\u2081 ist keine ganze Zahl! \n"
            self.gui.entered_w_1.set(str(self.default_WEIGHT_1))

        if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            try:
                self.initialized_weight_2 = int(self.gui.entered_w_2.get())
            except:
                error_msg += "Das eingegebene Gewicht w\u2082 ist keine ganze Zahl! \n"
                self.gui.entered_w_2.set(str(self.default_WEIGHT_2))

        try:
            self.initialized_threshold = int(self.gui.entered_threshold.get())
        except:
            if selected_mode == self.mode_LINEAR_CLASSIFICATION:
                error_msg += "Der eingegebene Schwellenwert ist keine ganze Zahl! \n"
            elif selected_mode == self.mode_LINEAR_REGRESSION:
                error_msg += "Das eingegebene Gewicht w\u2080 ist keine ganze Zahl! \n"
            self.gui.entered_threshold.set(str(self.default_THRESHOLD))

        # Prüfen, ob die Lernrate ausgewählt ist.
        # Falls nicht, werden die Default-Werte verwendet
        # Andernfalls wird die Eingabe auf die Eigenschaft "positive Zahl" hin überprüft
        # Tritt beim Ausführen ein Fehler auf, werden die Default-Werte verwendet
        if self.gui.use_learning_rate.get() == 0:
            if selected_mode == self.mode_LINEAR_CLASSIFICATION:
                self.initialized_learning_rate = self.default_LEARNING_RATE_LINEAR_CLASSIFICATION
                self.gui.entered_learning_rate.set(self.default_LEARNING_RATE_LINEAR_CLASSIFICATION)
            elif selected_mode == self.mode_LINEAR_REGRESSION:
                self.initialized_learning_rate = self.default_LEARNING_RATE_LINEAR_REGRESSION
                self.gui.entered_learning_rate.set(self.default_LEARNING_RATE_LINEAR_REGRESSION)
        else:
            positive_entry = True
            try:
                self.initialized_learning_rate = float(self.gui.entered_learning_rate.get())

                if self.initialized_learning_rate <= 0:
                    positive_entry = False
                    raise Exception()

                if self.initialized_learning_rate.is_integer():
                    self.initialized_learning_rate = int(self.initialized_learning_rate)
            except:
                if not positive_entry:
                    error_msg += "Die eingegebene Lernrate muss positiv sein!"
                else:
                    error_msg += "Die eingegebene Lernrate ist keine Zahl!"
                if selected_mode == self.mode_LINEAR_CLASSIFICATION:
                    self.initialized_learning_rate = self.default_LEARNING_RATE_LINEAR_CLASSIFICATION
                    self.gui.entered_learning_rate.set(self.default_LEARNING_RATE_LINEAR_CLASSIFICATION)
                elif selected_mode == self.mode_LINEAR_REGRESSION:
                    self.initialized_learning_rate = self.default_LEARNING_RATE_LINEAR_REGRESSION
                    self.gui.entered_learning_rate.set(self.default_LEARNING_RATE_LINEAR_REGRESSION)
        return error_msg

    def call_initialize_perceptron(self):
        """
        Initialisiert das Perzeptron mit den eingegebenen Werten. Zuerst werden die Eingaben überprüft und dann die
        Initialisierung in Abhängigkeit vom aktuell gewählten Modus aufgerufen. Die Initialisierung setzt das Perzeptron
        in jedem Fall zurück. Nach erfolgreicher Initialisierung ist das Perzeptron initialisiert, andernfalls None
        """
        # Speichern des aktuell ausgewählten Modus
        current_mode = self.gui.selected_mode.get()

        # Überprüfen der Eingaben für die Parameter des Perzeptrons
        input_errors = self.check_inputs(current_mode)

        if current_mode == self.mode_LINEAR_CLASSIFICATION:
            self.initialize_perceptron_mode_linear_classification(error_msg=input_errors)
        elif current_mode == self.mode_LINEAR_REGRESSION:
            self.initialize_perceptron_mode_linear_regression(error_msg=input_errors)

    def initialize_perceptron_mode_linear_classification(self, error_msg):
        self.current_perceptron = None
        self.initialized_with_learning_rate = False
        self.index_of_dataset_to_train = 0

        if error_msg == "":

            self.current_perceptron = perceptron([self.initialized_weight_1, self.initialized_weight_2],
                                                 self.initialized_threshold, self.initialized_learning_rate)
            if self.gui.use_learning_rate.get() == 1:
                self.current_perceptron.initialized_with_learning_rate = True

            # Aktualisierung der Perzeptron-Anzeige
            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)
            self.gui.current_training_steps.set(self.current_perceptron.number_of_training_steps)

            self.build_explanation(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                   current_perceptron=self.current_perceptron)
            self.build_training_protocol(clear_explanation=True)

            # Perzeptron korrekt geladen und es liegen bereits Daten vor
            if self.data_loaded:
                self.gui.update_progressbar(
                    self.current_perceptron.number_of_correct_classified(self.read_in_train_data),
                    self.size_training_data)

                # Replot regelt hier die Anzeige für den Standardfall, wenn Daten vorhanden und Perzeptron
                # erfolgreich initialisiert wurde
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                                emphasize_point=False, data_point=self.read_in_train_data[self.index_of_dataset_to_train])

            # Perzeptron geladen, aber es liegen keine Trainingsdaten vor
            else:
                self.gui.update_progressbar(0, 1)

                # Sicherheitshalber(kann evtl. weg), falls man durch Klick nach der Fehlermeldung und noch geöffnetem
                # Drop-Down die Gewichtsaktualisierung auswählt
                # Diese soll bei nicht geladenen Daten nicht möglich sein
                # Zurücksetzen der Auswahl
                callback_func = self.gui.selected_display_mode_for_linear_classification.trace_info()[0]
                self.gui.selected_display_mode_for_linear_classification.trace_remove(callback_func[0],
                                                                                      callback_func[1])
                self.gui.selected_display_mode_for_linear_classification.set("Trenngerade")
                self.gui.selected_display_mode_for_linear_classification.trace_add(mode="write",
                                                                                   callback=self.call_display_mode_switched_linear_classification)
                self.gui.show_next_point_to_train.set(0)
                self.gui.adapt_plot_contol_options(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                   display_mode="Trenngerade")

                # Anzeige der Trenngerade, welche das Perzeptron symbolisiert
                self.gui.reset_plotting_figure("2D")
                my_Plotting.plotting_separation_line(self.gui.plot2D, self.current_perceptron.weights,
                                                     self.current_perceptron.threshold, data_loaded=False)
                if self.gui.show_half_area.get() == 1:
                    my_Plotting.fill(self.gui.plot2D, self.current_perceptron.weights,
                                     self.current_perceptron.threshold, data_loaded=False)
                my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                self.gui.canvas.draw()

        # error_msg war nicht leer, d.h. es gab also einen Fehler beim Einlesen. Perzeptron ist somit None
        # Zurücksetzen der Anzeige
        else:
            # Aktualisierung der Perzeptron-Anzeige
            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)
            self.gui.current_training_steps.set(0)
            self.gui.update_progressbar(0, 1)

            self.build_explanation(current_mode=self.mode_LINEAR_CLASSIFICATION, clear_explanation=True)
            self.build_training_protocol(clear_explanation=True)

            # Zurücksetzen des Plots
            self.gui.reset_plotting_figure("2D")
            if self.data_loaded:
                my_Plotting.plotting_data(plt1=self.gui.plot2D, headerinfo=self.read_in_train_data_header,
                                          dataset=self.read_in_train_data, acual_mode=self.mode_LINEAR_CLASSIFICATION)
            my_Plotting.plotting_axis_arrows(self.gui.plot2D)
            self.gui.canvas.draw()

            callback_func = self.gui.selected_display_mode_for_linear_classification.trace_info()[0]
            self.gui.selected_display_mode_for_linear_classification.trace_remove(callback_func[0],
                                                                                  callback_func[1])
            self.gui.selected_display_mode_for_linear_classification.set("Trenngerade")
            self.gui.selected_display_mode_for_linear_classification.trace_add(mode="write",
                                                                               callback=self.call_display_mode_switched_linear_classification)

            self.gui.show_next_point_to_train.set(0)
            self.gui.show_half_area.set(0)

            self.gui.adapt_plot_contol_options(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                               display_mode="Trenngerade")

            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    def initialize_perceptron_mode_linear_regression(self, error_msg):
        self.current_perceptron = None
        self.initialized_with_learning_rate = False
        if error_msg == "" and self.data_loaded:
            self.current_perceptron = perceptron([self.initialized_weight_1],
                                                 self.initialized_threshold, self.initialized_learning_rate)

            if self.gui.use_learning_rate.get() == 1:
                self.current_perceptron.initialized_with_learning_rate = True

            # Aktualisierung der Perzeptron-Anzeige
            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)

            self.gui.current_training_steps.set(self.current_perceptron.number_of_training_steps)

            # Replot regelt hier die Anzeige für den Standardfall, wenn Daten vorhanden und Perzeptron
            # erfolgreich initialisiert wurde
            self.gui.replot(header_data=self.read_in_train_data_header,
                            training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)

            self.build_explanation(current_mode=self.mode_LINEAR_REGRESSION,
                                   current_perceptron=self.current_perceptron)


        # Design-Entscheidung: Perzeptron kann für die lineare Regression nur initialisiert werden, wenn Daten geladen
        # wurden
        # error_msg nicht leer oder Daten nicht geladen
        else:

            # Aktualisierung der Perzeptron-Anzeige, Perzeptron ist None
            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)
            self.gui.current_training_steps.set(0)

            self.build_explanation(current_mode=self.mode_LINEAR_REGRESSION, clear_explanation=True)

            # Sicherheitshalber siehe Methode für lineare Klassifikation(kann evtl. weg)
            callback_func = self.gui.selected_display_mode_for_linear_regression.trace_info()[0]
            self.gui.selected_display_mode_for_linear_regression.trace_remove(callback_func[0],
                                                                              callback_func[1])
            self.gui.selected_display_mode_for_linear_regression.set("Geradenansicht")
            self.gui.selected_display_mode_for_linear_regression.trace_add(mode="write",
                                                                           callback=self.call_display_mode_switched_linear_regression)
            self.gui.reset_plotting_figure("2D")

            # Keine Initialisierung aufgrund fehlerhafter Parameter
            if self.data_loaded:
                my_Plotting.plotting_data(self.gui.plot2D, self.read_in_train_data_header,
                                          self.read_in_train_data, self.gui.selected_mode.get())
            else:
                error_msg += "Keine Daten geladen! Perzeptron wurde nicht initialisiert!\n"

            my_Plotting.plotting_axis_arrows(self.gui.plot2D)
            self.gui.canvas.draw()

            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    def call_reset_perceptron(self):
        """
        Setzt das Perzeptron auf den Zustand der Initialisierung zurück.
        Aktualisiert auch die Anzeige des Perzeptrons, den Plot, die Anzahl der Trainingsschritte und
        ggf. die ProgressBar
        """
        selected_mode = self.gui.selected_mode.get()

        self.current_perceptron.number_of_training_steps = 0
        self.gui.current_training_steps.set(0)

        self.index_of_dataset_to_train = 0

        # Zurücksetzen der Perzeptronwerte auf die Werte der letzten Initialisierung
        if self.data_loaded and self.current_perceptron is not None:

            self.current_perceptron.number_of_training_steps = 0

            if selected_mode == self.mode_LINEAR_CLASSIFICATION:
                self.current_perceptron.weights = [self.initialized_weight_1, self.initialized_weight_2]
                self.current_perceptron.threshold = self.initialized_threshold
                self.current_perceptron.weights_history = [
                    [self.current_perceptron.weights[0], self.current_perceptron.weights[1],
                     self.current_perceptron.threshold]]
                self.gui.update_progressbar(
                    self.current_perceptron.number_of_correct_classified(self.read_in_train_data),
                    self.size_training_data)
                # Design-Entscheidung: Beim Zurücksetzen werden auch die akutellen Perzeptronwerte in die Eingabefelder
                # geschrieben
                self.gui.entered_w_1.set(self.initialized_weight_1)
                self.gui.entered_w_2.set(self.initialized_weight_2)
                self.gui.entered_threshold.set(self.initialized_threshold)
            elif selected_mode == self.mode_LINEAR_REGRESSION:
                self.current_perceptron.weights = [self.initialized_weight_1]
                self.current_perceptron.threshold = self.initialized_threshold
                self.current_perceptron.weights_history = [
                    [self.current_perceptron.weights[0], self.current_perceptron.threshold]]
                self.gui.entered_epochs.set(self.default_EPOCHS)
                # Design-Entscheidung: Beim Zurücksetzen werden auch die akutellen Perzeptronwerte in die Eingabefelder
                # geschrieben
                self.gui.entered_w_1.set(self.initialized_weight_1)
                self.gui.entered_threshold.set(self.initialized_threshold)

            self.current_perceptron.learning_rate = self.initialized_learning_rate
            self.gui.entered_learning_rate.set(self.initialized_learning_rate)

            self.gui.replot(header_data=self.read_in_train_data_header,
                            training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                            emphasize_point=False, data_point=self.read_in_train_data[self.index_of_dataset_to_train])

            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)

            self.build_explanation(current_mode=self.gui.selected_mode.get(),
                                   current_perceptron=self.current_perceptron)
            self.build_training_protocol(clear_explanation=True)
        else:
            error_msg = ""
            if not self.data_loaded:
                error_msg += "Keine Trainingsdaten geladen.\n"
            if self.current_perceptron is None:
                error_msg += "Das Perzeptron ist nicht initialisiert."
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    # NICHT VERWENDET
    def call_train_all(self):
        """
        Trainiert im Fall Modus "Lineare Klassifikation" einmal jeden Trainingsdatenpunkt.
        Im Fall Modus "Lineare Regression" werden so viele Trainingsschritte gemacht, wie in Epoche angegeben. Dabei
        wird ein Trainingsschritt mit dem Gesamtfehler der Trainingsdaten durchgeführt.
        Im Anschluss werden die Anzeigen aktualisiert. Im Modus "Lineare Klassifikation" wird der Index des als
        nächstes zu trainierenden Trainingsdatums auf das erste Element gesetzt
        """
        if self.data_loaded and self.current_perceptron is not None:
            # Fallunterscheidung je gewähltem Modus
            # Trainiert einmal jedes Trainingsdatum
            if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
                self.current_perceptron.train_all_classification(self.read_in_train_data)
                # Setzt den Index des als nächstes zu trainierenden Datums auf das erste Element
                self.index_of_dataset_to_train = 0
                self.gui.update_progressbar(
                    self.current_perceptron.number_of_correct_classified(self.read_in_train_data),
                    self.size_training_data)
            # Trainiert so viele Trainingsschritte mit dem Gesamtfehler, wie in Epochen angegeben
            elif self.gui.selected_mode.get() == self.mode_LINEAR_REGRESSION:
                # Einlesen und prüfen der Eingabe für Epochen
                # Ausgabe eine Fehlermeldung bei fehlerhafter Epochenzahl
                negative_entry = False
                try:
                    epochs = int(self.gui.entered_epochs.get())
                    if epochs <= 0:
                        negative_entry = True
                        raise Exception()
                    # Trainieren des Perzeptrons mit dem Gesamtfehler im Modus "Lineare Regression" Anzahl der Epochen mal
                    self.current_perceptron.train_linear_regession(self.read_in_train_data, epochs)
                except:
                    error_msg = ""
                    if negative_entry:
                        error_msg = "Die eingegebene Epochenzahl muss eine positive ganze Zahl sein!"
                    else:
                        error_msg = "Die eingegebene Epochenzahl keine ganze Zahl!"
                    epochs = self.default_EPOCHS
                    self.gui.entered_epochs.set(epochs)
                    messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)
            # Ende der Fallunterscheidung
            # Aktualsisiert den Plot, die Perzeptronanzeige, die Anzeige der Geradengleichung und die Anzahl der
            # Trainingsschritte
            self.gui.replot(header_data=self.read_in_train_data_header,
                            training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)
            self.gui.update_perceptron_visualization(False, self.current_perceptron)
            self.gui.lbl_explanation_regression.config(
                text=self.build_explanation_string_general(self.current_perceptron), anchor='nw')

            self.gui.current_training_steps.set(self.current_perceptron.number_of_training_steps)
        else:
            error_msg = ""
            if not self.data_loaded:
                error_msg += "Keine Trainingsdaten geladen.\n"
            if self.current_perceptron is None:
                error_msg += "Das Perzeptron ist nicht initialisiert."
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    def running_method(self):
        """
        Methode, welche den Trainingsschritt während des automatischen Trainings realisiert und die Pause umsetzt.
        """
        # Da diese Methode nur in call_train_automode aufgerufen wird, kann davon ausgegangen werden, dass das
        # an dieser Stelle initialisiert ist und Daten geladen sind
        # Des Weiteren sind wir sicher im Modus "Lineare Klassifikation."

        dataset_to_train = self.read_in_train_data[self.index_of_dataset_to_train]

        # Berechnet die Ausgabe des Perzeptrons vor dem Trainingsschritt
        result_query = self.current_perceptron.calc_output_perceptron_classification(
            dataset_to_train[0:len(dataset_to_train) - 1])

        # Speichert die Gewichte und den Schwellenwert vor dem Trainingsschritt
        temp_w_1 = self.current_perceptron.weights[0]
        temp_w_2 = self.current_perceptron.weights[1]
        temp_threshold = self.current_perceptron.threshold

        # Trainiert das Perzeptron mit dem aktuellen Trainingsdatum dataset_to_train
        self.current_perceptron.train_single_dataset(dataset_to_train)

        # Aktualisiert den Index des als Nächstes zu trainierenden Datums
        self.index_of_dataset_to_train = (self.index_of_dataset_to_train + 1) % (len(self.read_in_train_data))

        # Aktualisiert die Anzahl der Trainingsschritte
        self.gui.current_training_steps.set(self.current_perceptron.number_of_training_steps)

        # Aktualisiert den Plot
        self.gui.replot(header_data=self.read_in_train_data_header,
                        training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                        emphasize_point=True,
                        data_point=dataset_to_train)
        self.gui.update_perceptron_visualization(True, self.current_perceptron, temp_w_1, temp_w_2,
                                                 temp_threshold,
                                                 dataset_to_train, result_query)
        self.gui.update_progressbar(
            self.current_perceptron.number_of_correct_classified(self.read_in_train_data),
            self.size_training_data)

        # Anzeige der Trenngerade
        self.build_explanation(current_mode=self.gui.selected_mode.get(),
                               current_perceptron=self.current_perceptron)

        # Anzeige des Trainingsprotokolls
        self.build_training_protocol(current_perceptron=self.current_perceptron,
                                     current_calculated_output=result_query,
                                     current_inputs=dataset_to_train[:-1], current_target=dataset_to_train[-1],
                                     old_weights=[temp_w_1, temp_w_2], old_threshold=temp_threshold)

        # Anapassen der Kontrollbuttons schneller/langsamer
        if self.auto_mode_running_speed >= 2000:
            self.gui.btn_train_automode_slower['state'] = tkinter.DISABLED
            self.gui.btn_train_automode_faster['state'] = tkinter.NORMAL
        elif self.auto_mode_running_speed <= 500:
            self.gui.btn_train_automode_faster['state'] = tkinter.DISABLED
            self.gui.btn_train_automode_slower['state'] = tkinter.NORMAL
        else:
            self.gui.btn_train_automode_faster['state'] = tkinter.NORMAL
            self.gui.btn_train_automode_slower['state'] = tkinter.NORMAL

        # Warten zwischen den Trainingsrunden
        # Speicherung der callId um bei Beeindigung des Trainings dieses direkt abbrechen zu können

        self.callID = self.gui.window.after(ms=self.auto_mode_running_speed, func=self.running_method)

    def call_auto_mode_running_speed_faster(self):
        """
        Erhöht die Geschwindigkeit des automatischen Trainings. Dafür wird der running_speed halbiert, welcher
        die Wartezeit zwischen den Iterationen darstellt
        """
        if self.auto_mode_running_speed == 2000:
            self.auto_mode_running_speed = 1000
        elif self.auto_mode_running_speed == 1000:
            self.auto_mode_running_speed = 500
        else:
            self.auto_mode_running_speed = 500

    def call_auto_mode_running_speed_slower(self):
        """
        Reduziert die Geschwindigkeit des automatischen Trainings. Dafür wird der running_speed verdoppelt, welcher
        die Wartezeit zwischen den Iterationen darstellt
        """
        if self.auto_mode_running_speed == 500:
            self.auto_mode_running_speed = 1000
        elif self.auto_mode_running_speed == 1000:
            self.auto_mode_running_speed = 2000
        else:
            self.auto_mode_running_speed = 2000

    def call_train_automode(self):
        """
        Startet bzw beendet das automatische Training, passt die Kontrollbuttons entsprechend an und
        sperrt die restliche Oberfläche
        Durch die Oberfläche ist sichergestellt, dass wir im Modus "Lineare Klassifikation" sind,
        die Klassifikation eines Punkts nicht aktiviert ist und die Anzeigeeinstellung "Trenngerade"
        ist
        """
        if self.data_loaded and self.current_perceptron is not None:

            # Beenden des automatischen Trainings
            if self.auto_training_running:
                self.auto_training_running = False

                # Sicherheitsmaßnahme, dass kein Trainingsprozess neu gestartet werden kann, bevor der alte sauber
                # beendet wurde
                self.gui.btn_train_auto_mode['state'] = tkinter.DISABLED

                # Beenden der automatischen Ausführung, falls momentan noch ein Aufruf am Warten ist
                if self.callID is not None:
                    self.gui.window.after_cancel(self.callID)
                    self.callID = None

                # Wechsel der Kontrollbuttons
                self.gui.adapt_perceptron_control_buttons(current_mode=self.gui.selected_mode.get(),
                                                          classification_of_a_point_enabled=self.gui.classification_point_enabled.get(),
                                                          auto_mode_running=False,
                                                          display_mode=self.gui.selected_display_mode_for_linear_classification.get())

                # Aktualisiert den Plot
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                                emphasize_point=True, data_point=self.read_in_train_data[
                        self.index_of_dataset_to_train - 1 % self.size_training_data])

                # Rückgängigmachen von Sicherheitsmaßnahme (siehe else-Teil)
                self.gui.ck_btn_show_half_areas['command'] = self.call_replot_show_half_areas
                self.gui.ck_btn_show_next_point_to_train['command'] = self.call_replot_show_next_point_to_train

                # Entsperren der Oberfläche
                self.gui.adapt_enable_disable_mode_selection(enable_mode_selection=True)
                self.gui.adapt_enable_disable_input_controls(enable_inputs=True)
                self.gui.adapt_plot_contol_options(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                   display_mode="Trenngerade", auto_mode_running=False)

                self.gui.ck_btn_classifiy_point['state'] = tkinter.NORMAL
                self.gui.btn_train_auto_mode['state'] = tkinter.NORMAL


            # Starten des automatischen Trainings
            else:
                self.auto_training_running = True
                self.auto_mode_running_speed = 1000

                # Sperren der Oberfläche
                self.gui.adapt_enable_disable_mode_selection(enable_mode_selection=False)
                self.gui.adapt_enable_disable_input_controls(enable_inputs=False)

                self.gui.adapt_plot_contol_options(current_mode=self.mode_LINEAR_CLASSIFICATION,
                                                   display_mode="Trenngerade", auto_mode_running=True)
                self.gui.ck_btn_classifiy_point['state'] = tkinter.DISABLED

                # Entfernen des sofortigen Replots, lediglich die entsprechende IntVar wird noch gesetzt
                # Wird erst beim nächsten Replot, d.h. beim nächsten Trainingssschritt aktiv.
                # Verhindert, das gleichzeitig ein Replot der Automode und ein Replot durch den Check-Button passiert
                self.gui.ck_btn_show_half_areas['command'] = ""
                self.gui.ck_btn_show_next_point_to_train['command'] = ""

                # Wechsel der Kontrollbuttons
                self.gui.adapt_perceptron_control_buttons(current_mode=self.gui.selected_mode.get(),
                                                          auto_mode_running=True,
                                                          display_mode=self.gui.selected_display_mode_for_linear_classification.get())

                # Starten der automatischen Ausführung
                self.running_method()

        else:
            error_msg = ""
            if not self.data_loaded:
                error_msg += "Keine Trainingsdaten geladen.\n"
            if self.current_perceptron is None:
                error_msg += "Das Perzeptron ist nicht initialisiert."
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    def call_train(self):
        """
        Trainiert ein einzelnes Datum. Wird deshalb nur im Modus "Lineare Klassifikation" verwendet.
        Trainiert das Perzeptron mit einem Trainingsdatum, aktualisiert die Anzeige des Plots und des Perzeptrons,
        aktualisiert die Trainingsschritte und der Progressbar sowie des Index des als Nächstes zu trainierenden Datums
        """
        if self.data_loaded and self.current_perceptron is not None:
            # Fallunterscheidung je gewähltem Modus
            if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
                dataset_to_train = self.read_in_train_data[self.index_of_dataset_to_train]

                # Berechnet die Ausgabe des Perzeptrons vor dem Trainingsschritt
                result_query = self.current_perceptron.calc_output_perceptron_classification(
                    dataset_to_train[0:len(dataset_to_train) - 1])

                # Speichert die Gewichte und den Schwellenwert vor dem Trainingsschritt
                temp_w_1 = self.current_perceptron.weights[0]
                temp_w_2 = self.current_perceptron.weights[1]
                temp_threshold = self.current_perceptron.threshold

                # Trainiert das Perzeptron mit dem aktuellen Trainingsdatum dataset_to_train
                self.current_perceptron.train_single_dataset(dataset_to_train)

                # Aktualisiert den Index des als Nächstes zu trainierenden Datums
                self.index_of_dataset_to_train = (self.index_of_dataset_to_train + 1) % (len(self.read_in_train_data))

                # Aktualisiert die Anzahl der Trainingsschritte
                self.gui.current_training_steps.set(self.current_perceptron.number_of_training_steps)

                # Aktualisiert den Plot
                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron,
                                emphasize_point=True,
                                data_point=dataset_to_train)
                self.gui.update_perceptron_visualization(True, self.current_perceptron, temp_w_1, temp_w_2,
                                                         temp_threshold,
                                                         dataset_to_train, result_query)
                self.gui.update_progressbar(
                    self.current_perceptron.number_of_correct_classified(self.read_in_train_data),
                    self.size_training_data)

                # Anzeige des Trainingsprotokolls
                self.build_training_protocol(current_perceptron=self.current_perceptron,
                                             current_calculated_output=result_query,
                                             current_inputs=dataset_to_train[:-1], current_target=dataset_to_train[-1],
                                             old_weights=[temp_w_1, temp_w_2], old_threshold=temp_threshold)

            elif self.gui.selected_mode.get() == self.mode_LINEAR_REGRESSION:
                # Einlesen und prüfen der Eingabe für Epochen
                # Ausgabe eine Fehlermeldung bei fehlerhafter Epochenzahl
                negative_entry = False
                try:
                    epochs = int(self.gui.entered_epochs.get())
                    if epochs <= 0:
                        negative_entry = True
                        raise Exception()
                    # Trainieren des Perzeptrons mit dem Gesamtfehler im Modus "Lineare Regression" Anzahl der Epochen mal
                    self.current_perceptron.train_linear_regession(self.read_in_train_data, epochs)
                except:
                    error_msg = ""
                    if negative_entry:
                        error_msg = "Die eingegebene Epochenzahl muss eine positive ganze Zahl sein!"
                    else:
                        error_msg = "Die eingegebene Epochenzahl keine ganze Zahl!"
                    epochs = self.default_EPOCHS
                    self.gui.entered_epochs.set(epochs)
                    messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

                self.gui.replot(header_data=self.read_in_train_data_header,
                                training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)
                self.gui.update_perceptron_visualization(False, self.current_perceptron)

            # Anzeige der Gleichung der Geraden(Trenn- oder Regressionsgerade)
            self.build_explanation(current_mode=self.gui.selected_mode.get(),
                                   current_perceptron=self.current_perceptron)
            self.gui.current_training_steps.set(self.current_perceptron.number_of_training_steps)
        else:
            error_msg = ""
            if not self.data_loaded:
                error_msg += "Keine Trainingsdaten geladen.\n"
            if self.current_perceptron is None:
                error_msg += "Das Perzeptron ist nicht initialisiert."
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    def call_display_classification(self):
        """
        Regelt die Anpassung der kompletten Oberfläche für die Klassifikation eines Punkts mit dem aktuellen Perzeptron.
        In Abhängigkeit vom ausgewählten Modus und der gewählten Anzeigeeinstellung,... werden die entsprechenden
        Schaltflächen ein- oder ausgeblendet. Methode wird nur im Modus "Lineare Klassifikation" verwendet.
        Methode für den Toggle-Button
        """
        # Vorbereiten der Klassifikation
        # Herstellen der Initialbelegung
        self.classification_phase = "phase_0"
        self.point_to_classify_x_1 = self.default_CLASSIFICATION_POINT_INPUT_X1
        self.point_to_classify_x_2 = self.default_CLASSIFICATION_POINT_INPUT_X2
        self.point_to_classify_target = self.default_CLASSIFICATION_POINT_TARGET
        self.gui.entered_classify_input_x_1.set(str(self.default_CLASSIFICATION_POINT_INPUT_X1))
        self.gui.entered_classify_input_x_2.set(str(self.default_CLASSIFICATION_POINT_INPUT_X2))
        self.gui.entered_classify_target.set(str(self.default_CLASSIFICATION_POINT_TARGET))

        selected_mode = self.gui.selected_mode.get()
        selected_display_mode = self.gui.selected_display_mode_for_linear_classification.get()

        # Voraussetzung das klassifiziert werden kann
        if self.current_perceptron is not None:
            self.gui.update_perceptron_visualization(display_training_step=False,
                                                     current_perceptron=self.current_perceptron)

            # Klassifikation aktiviert
            if self.gui.classification_point_enabled.get() == 1:
                self.gui.adapt_enable_disable_input_controls(enable_inputs=False)
                self.gui.adapt_perceptron_control_buttons(current_mode=selected_mode,
                                                          classification_of_a_point_enabled=1)
                self.gui.adapt_plot_contol_options(current_mode=selected_mode,
                                                   classification_of_a_point_enabled=1,
                                                   display_mode=selected_display_mode)
                self.gui.adapt_classification_of_a_point_control(current_mode=selected_mode,
                                                                 display_mode=selected_display_mode,
                                                                 classification_of_a_point_enabled=1,
                                                                 phase_of_classification=self.classification_phase)

                self.classification_phase = "phase_1"
                # Sowohl Perzeptron als auch Daten sind geladen
                if self.data_loaded:
                    self.gui.replot(header_data=self.read_in_train_data_header,
                                    training_data=self.read_in_train_data,
                                    current_perceptron=self.current_perceptron)

                # Es sind momentan keine Trainingsdaten geladen
                # Damit wird nur die das Perzeptron repräsentierende Trenngerade angezeigt
                else:
                    self.gui.clear_plot()
                    if self.gui.show_half_area.get() == 1:
                        my_Plotting.fill(self.gui.plot2D, self.current_perceptron.weights,
                                         self.current_perceptron.threshold,
                                         data_loaded=False)
                    my_Plotting.plotting_separation_line(self.gui.plot2D, self.current_perceptron.weights,
                                                         self.current_perceptron.threshold, data_loaded=False)

                    my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                    self.gui.canvas.draw()

            # Klassifikation deaktiviert
            elif self.gui.classification_point_enabled.get() == 0:
                self.gui.adapt_enable_disable_input_controls(enable_inputs=True)
                self.gui.adapt_perceptron_control_buttons(current_mode=selected_mode,
                                                          classification_of_a_point_enabled=0)
                self.gui.adapt_plot_contol_options(current_mode=selected_mode,
                                                   classification_of_a_point_enabled=0,
                                                   display_mode=selected_display_mode)
                self.gui.adapt_classification_of_a_point_control(current_mode=selected_mode,
                                                                 display_mode=selected_display_mode,
                                                                 classification_of_a_point_enabled=0,
                                                                 phase_of_classification=self.classification_phase)

                # Ausschalten der Klassifikation
                # Falls der nächste zu trainierende Punkt vorher ausgewählt war, wird er wieder angezeigt
                # Halbräume brauchen keine Zusatzinfo; ob angezeigt wird über die Variable in gui abgefragt
                if self.data_loaded:
                    self.gui.replot(header_data=self.read_in_train_data_header,
                                    training_data=self.read_in_train_data, current_perceptron=self.current_perceptron)

                else:
                    self.gui.clear_plot()
                    if self.gui.show_half_area.get() == 1:
                        my_Plotting.fill(self.gui.plot2D, self.current_perceptron.weights,
                                         self.current_perceptron.threshold,
                                         data_loaded=False)
                    my_Plotting.plotting_separation_line(self.gui.plot2D, self.current_perceptron.weights,
                                                         self.current_perceptron.threshold, data_loaded=False)
                    my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                    self.gui.canvas.draw()
        else:
            self.gui.classification_point_enabled.set(0)
            error_msg = "Das Perzeptron ist nicht initialisiert."
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    def call_classification(self):
        """
        Regelt die Klassifikationslogik. Methode für den Button, der in der Klassifikationsöberfläche angezeift wird
        :return:
        """
        # Einlesen des zu klassifizierenden Datenpunkts und dessen erwartetes Label
        error_msg = ""

        # Einlesen und Speichern der eingegebenen Werte. Falls ein Fehler auftritt, wird der Fehlermeldungstext ergänzt
        if self.classification_phase == "phase_2":
            try:
                self.point_to_classify_x_1 = int(self.gui.entered_classify_input_x_1.get())
            except:
                error_msg += "Der x\u2081 Wert ist keine ganze Zahl! \n"
                self.point_to_classify_x_1 = self.default_CLASSIFICATION_POINT_INPUT_X1
                self.gui.entered_classify_input_x_1.set(str(self.point_to_classify_x_1))

            try:
                self.point_to_classify_x_2 = int(self.gui.entered_classify_input_x_2.get())
            except:
                error_msg += "Der x\u2082  Wert ist keine ganze Zahl! \n"
                self.point_to_classify_x_2 = self.default_CLASSIFICATION_POINT_INPUT_X2
                self.gui.entered_classify_input_x_2.set(str(self.point_to_classify_x_2))

            try:
                read_in = int(self.gui.entered_classify_target.get())
                if read_in == 0 or read_in == 1:
                    self.point_to_classify_target = read_in
                else:
                    raise Exception()
            except:
                error_msg += "Das eingegebene Label muss 0 oder 1 sein! \n"
                self.point_to_classify_target = self.default_CLASSIFICATION_POINT_TARGET
                self.gui.entered_classify_target.set(str(self.point_to_classify_target))

        # Wenn der Fehlermeldungstext leer ist, wurden korrekte Eingaben eingelesen
        if error_msg == "":

            if self.classification_phase == "phase_1":
                self.gui.update_perceptron_visualization(display_training_step=True,
                                                         current_perceptron=self.current_perceptron,
                                                         current_state="classification")
                if self.data_loaded:
                    self.gui.replot(header_data=self.read_in_train_data_header,
                                    training_data=self.read_in_train_data,
                                    current_perceptron=self.current_perceptron)
                # Keine Daten geladen
                else:
                    self.gui.clear_plot()
                    if self.gui.show_half_area.get() == 1:
                        my_Plotting.fill(self.gui.plot2D, self.current_perceptron.weights,
                                         self.current_perceptron.threshold,
                                         data_loaded=False)
                    my_Plotting.plotting_separation_line(self.gui.plot2D, self.current_perceptron.weights,
                                                         self.current_perceptron.threshold, data_loaded=False)

                    my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                    self.gui.canvas.draw()


            elif self.classification_phase == "phase_2" or \
                    self.classification_phase == "phase_3" or \
                    self.classification_phase == "phase_0":

                if self.data_loaded:
                    self.gui.replot(header_data=self.read_in_train_data_header,
                                    training_data=self.read_in_train_data,
                                    current_perceptron=self.current_perceptron,
                                    point_to_classify=[[self.point_to_classify_x_1, self.point_to_classify_x_2],
                                                       self.point_to_classify_target])
                # Keine Daten geladen
                else:
                    self.gui.clear_plot()
                    if self.gui.show_half_area.get() == 1:
                        my_Plotting.fill(self.gui.plot2D, self.current_perceptron.weights,
                                         self.current_perceptron.threshold,
                                         data_loaded=False)
                    my_Plotting.plotting_separation_line(self.gui.plot2D, self.current_perceptron.weights,
                                                         self.current_perceptron.threshold, data_loaded=False)

                    my_Plotting.plot_point(self.gui.plot2D, [self.point_to_classify_x_1, self.point_to_classify_x_2],
                                           self.point_to_classify_target)

                    my_Plotting.plotting_axis_arrows(self.gui.plot2D)
                    self.gui.canvas.draw()

                if self.classification_phase == "phase_2":
                    self.gui.update_perceptron_visualization(display_training_step=True,
                                                             current_perceptron=self.current_perceptron,
                                                             training_data_point=[self.point_to_classify_x_1,
                                                                                  self.point_to_classify_x_2,
                                                                                  self.point_to_classify_target],
                                                             current_state="classification")
                elif self.classification_phase == "phase_3" or self.classification_phase == "phase_0":
                    # Berechnung der Ausgabe mit dem aktuellen Perzeptron
                    # Die Klassifikationsoberfläche ist nur anzeigbar, wenn das Perzeptron initialisiert ist
                    result_query = self.current_perceptron.calc_output_perceptron_classification(
                        [self.point_to_classify_x_1, self.point_to_classify_x_2])
                    self.gui.update_perceptron_visualization(display_training_step=True,
                                                             current_perceptron=self.current_perceptron,
                                                             training_data_point=[self.point_to_classify_x_1,
                                                                                  self.point_to_classify_x_2,
                                                                                  self.point_to_classify_target],
                                                             calculated_output=result_query,
                                                             current_state="classification")

            self.gui.adapt_classification_of_a_point_control(current_mode=self.gui.selected_mode.get(),
                                                             display_mode=self.gui.selected_display_mode_for_linear_classification.get(),
                                                             classification_of_a_point_enabled=self.gui.classification_point_enabled.get(),
                                                             phase_of_classification=self.classification_phase)

            # Weiterschalten der Phase
            if self.classification_phase == "phase_0":
                self.classification_phase = "phase_1"
            elif self.classification_phase == "phase_1":
                self.classification_phase = "phase_2"
            elif self.classification_phase == "phase_2":
                self.classification_phase = "phase_3"
            else:
                self.classification_phase = "phase_0"

        # Falls beim Einlesen ein Fehler passiert ist
        else:
            # Verbleiben in der Phase zu Anzeige der Einlesemöglichkeit
            self.classification_phase = "phase_1"
            messagebox.showerror(title="Fehler", message=error_msg, parent=self.gui.window)

    def build_training_protocol(self, current_perceptron=None,
                                current_calculated_output=None, current_inputs=None,
                                current_target=None, old_weights=None, old_threshold=None,
                                clear_explanation=False):
        """
        Erzeugt das Trainingsprotokoll. Kann nur im Modus "Lineare Klassifikation" aufgerufen werden
        :param current_perceptron:
        :param current_calculated_output:
        :param current_inputs:
        :param current_target:
        :param old_weights:
        :param old_threshold:
        :param clear_explanation:
        :return:
        """
        if clear_explanation or current_perceptron is None:
            text_training_protocol = ""
        else:
            delta = current_target - current_calculated_output
            text_error_delta = "Berechnung des Fehlers \u03b4: \n" \
                               + "\u03b4 = Erwartete Ausgabe - Berechnete Ausgabe = " \
                               + str(int(current_target)) + " - " + str(int(current_calculated_output)) + " = " + str(
                int(delta))
            text_empty_line = " \n \n"
            text_weight_updates = "Aktualisierung der Gewichte: \n"
            if delta == 0:
                text_weight_updates += "w\u2081 := " + str(round(current_perceptron.weights[0], 4)) + "\n" \
                                       + "w\u2082 := " + str(round(current_perceptron.weights[1], 4)) + "\n" \
                                       + "\u03B8 := " + str(round(current_perceptron.threshold, 4))
            else:
                if delta == 1:
                    weight_calc_sign = " + "
                    threshold_clac_sign = " - "
                else:
                    weight_calc_sign = " - "
                    threshold_clac_sign = " + "

                text_weight_updates += \
                    "w\u2081 := w\u2081" + weight_calc_sign
                if current_perceptron.initialized_with_learning_rate:
                    text_weight_updates += "\u03b1\u00b7"
                text_weight_updates += "x\u2081 = " + str(round(old_weights[0], 4)) + weight_calc_sign
                if current_perceptron.initialized_with_learning_rate:
                    text_weight_updates += str(current_perceptron.learning_rate) + "\u00b7"
                if current_inputs[0] < 0:
                    text_weight_updates += "(" + str(current_inputs[0]) + ")"
                else:
                    text_weight_updates += str(current_inputs[0])
                text_weight_updates += " = " + str(round(current_perceptron.weights[0], 4)) + "\n"

                text_weight_updates += \
                    "w\u2082 := w\u2082" + weight_calc_sign
                if current_perceptron.initialized_with_learning_rate:
                    text_weight_updates += "\u03b1\u00b7"
                text_weight_updates += "x\u2082 = " + str(round(old_weights[1], 4)) + weight_calc_sign
                if current_perceptron.initialized_with_learning_rate:
                    text_weight_updates += str(current_perceptron.learning_rate) + "\u00b7"
                if current_inputs[1] < 0:
                    text_weight_updates += "(" + str(current_inputs[1]) + ")"
                else:
                    text_weight_updates += str(current_inputs[1])
                text_weight_updates += " = " + str(round(current_perceptron.weights[1], 4)) + "\n"

                text_weight_updates += \
                    "\u03B8 := \u03B8" + threshold_clac_sign
                if current_perceptron.initialized_with_learning_rate:
                    text_weight_updates += "\u03b1\u00b7"
                text_weight_updates += "1 = " + str(round(old_threshold, 4)) + threshold_clac_sign
                if current_perceptron.initialized_with_learning_rate:
                    text_weight_updates += str(current_perceptron.learning_rate) + "\u00b7"
                text_weight_updates += "1 = " + str(round(current_perceptron.threshold, 4)) + "\n"

            text_training_protocol = text_error_delta + text_empty_line + text_weight_updates

        self.gui.lbl_explanation_training_protocol.config(text=text_training_protocol, anchor='nw')

    def build_explanation(self, current_mode, current_perceptron=None, clear_explanation=False):
        """
        Methode, welche die Anzeige des Trainingsprotokolls bzw. der Geradengleichungen regelt. Falls clear_explanantion
        True gesetzt ist, wird die Anzeige gelöscht
        :param current_mode: aktuell gewählter Modus (Lineare Klassifikation, Lineare Regression)
        :param current_perceptron: aktuelles Perzeptron
        :param clear_explanation: Falls True, werden die Erklärungen gelöscht
        """
        if current_mode == self.mode_LINEAR_CLASSIFICATION:
            if clear_explanation or current_perceptron is None:
                text_separation_line = ""
            else:
                # Erstellen des Texts für die Gleichung der Trenngeraden
                if current_perceptron.weights[1] < 0:
                    correct_calc_sign = "- "
                else:
                    correct_calc_sign = "+ "
                text_separation_line = "Die Gleichung der Trenngerade lautet momentan:\n " + str(round(
                    current_perceptron.weights[0], 4)) + "x\u2081 " + correct_calc_sign + str(round(abs(
                    current_perceptron.weights[1]), 4)) + "x\u2082  = " + str(round(current_perceptron.threshold, 4))

            self.gui.lbl_explanation_separation_line.config(text=text_separation_line, anchor='nw')

        elif current_mode == self.mode_LINEAR_REGRESSION:
            if clear_explanation or current_perceptron is None:
                text_regression_line = ""
            else:
                if current_perceptron.threshold < 0:
                    correct_calc_sign = "- "
                else:
                    correct_calc_sign = "+ "
                text_regression_line = "Die Gleichung der Regressionsgerade lautet momentan: \n y = " + str(
                    current_perceptron.weights[0]) + "x\u2081 " + correct_calc_sign \
                                       + str(abs(current_perceptron.threshold))

            self.gui.lbl_explanation_regression.config(text=text_regression_line, anchor='nw')

    # NICHT VERWENDET
    def build_explanation_string_general(self, current_perc):
        """
        Liefert einen Text für die Trenn- bzw. Regressionsgerade des akutellen Perzeptrons
        :param current_perc: akutelles Perzeptron
        :return: beschreibender Text mit der Geradengleichung
        """
        explanation = ""
        actual_functional_equation = ""
        if self.gui.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            if current_perc.weights[1] < 0:
                correct_calc_sign = "- "
            else:
                correct_calc_sign = "+ "
            actual_functional_equation = "Die Gleichung der Trenngerade lautet momentan: " + str(
                current_perc.weights[0]) + "x\u2081 " + correct_calc_sign + str(abs(
                current_perc.weights[1])) + "x\u2082  = " + str(current_perc.threshold)
        elif self.gui.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            actual_functional_equation = "Die Gleichung der Regressionsgerade lautet momentan: \n y = " + str(
                current_perc.weights[0]) + "x\u2081 + " + str(current_perc.threshold)
        explanation = actual_functional_equation

        return explanation

    # NICHT VERWENDET
    # Builds explanation for example for the initialization for one training round
    def build_explanation_string_training_round(self, w_x, w_y, b, datapoint, calc_target):
        explanation = ""
        explanation_general = self.build_explanation_string_general(self.current_perceptron)
        explanation_training_round = "Der aktuelle Traininsdatensatz hat als Eingabe (" + str(
            datapoint[0]) + "|" + str(datapoint[1]) + ") und hat als erwartete Ausgabe den Wert: " + str(
            datapoint[2]) + "\n" + "Das Perzeptron hat die folgende Ausgabe berechnet: " + str(calc_target)
        if datapoint[2] == calc_target:
            explanation_adjust = "Da die erwartete Ausgabe mit der brechneten Ausgabe übereinstimmt, waren keine " \
                                 "Anpassungen noetig! "
        else:
            explanation_adjust = "Da die erwartete Ausgabe nicht mit der brechneten Ausgabe übereinstimmt, " \
                                 "waren Anpassungen noetig! "
        explanation = explanation_general + "\n" + explanation_training_round + "\n" + explanation_adjust
        return explanation


    def start_simulator(self):
        self.gui.window.mainloop()



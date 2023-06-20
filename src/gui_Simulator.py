import os
import tkinter as tk
import tkinter.ttk as ttk

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from src.my_Plotting import my_Plotting


class gui_Simulator:
    mode_LINEAR_CLASSIFICATION = "Lineare Klassifikation"
    mode_LINEAR_REGRESSION = "Lineare Regression"
    mode_LINEAR_CLASSIFICATION_SEPARATION_LINE = "Trenngerade"
    mode_LINEAR_CLASSIFICATION_WEIGHT_UPDATES = "Gewichtsaktualisierung"
    mode_LINEAR_REGRESSION_REGRESSION_LINE = "Geradenansicht"
    mode_LINEAR_REGRESSION_GRADIENT_DESCENT = "Gradientenabstieg"

    WINDOW_WIDTH_CLASSIFICATION = 1103
    WIDOW_HEIGHT_CLASSIFICATION = 824

    WINDOW_WIDTH_REGRESSION = 1103
    WIDOW_HEIGHT_REGRESSION = 710

    def __init__(self, m):
        self.main_object = m

        self.window = tk.Tk()
        self.window.title("Perzeptron Simulator - Beta 1.4.1")
        # nt für Windows
        if os.name == 'nt':
            self.window.iconphoto(True, tk.PhotoImage(file=".\Grafiken\Icon_simple_1.png"))
            self.img_perceptron_linear_classification = tk.PhotoImage(file=".\Grafiken\Abb_Perzeptron_new.png")
            self.img_perceptron_linear_regression = tk.PhotoImage(file=".\Grafiken\Abb_Perzeptron_new_lr.png")
            self.img_logo_university_1 = tk.PhotoImage(file=".\Grafiken\logo_fuchs_wolf.png")
        # posix für Linux und MacOS
        else:
            self.window.iconphoto(True, tk.PhotoImage(file="./Grafiken/Icon_simple_1.png"))
            self.img_perceptron_linear_classification = tk.PhotoImage(file="./Grafiken/Abb_Perzeptron_new.png")
            self.img_perceptron_linear_regression = tk.PhotoImage(file="./Grafiken/Abb_Perzeptron_new_lr.png")
            self.img_logo_university_1 = tk.PhotoImage(file="./Grafiken/logo_fuchs_wolf.png")

        # set minimum window size value
        self.window.minsize(self.WINDOW_WIDTH_CLASSIFICATION, self.WIDOW_HEIGHT_CLASSIFICATION)

        # set maximum window size value
        self.window.maxsize(self.WINDOW_WIDTH_CLASSIFICATION, self.WIDOW_HEIGHT_CLASSIFICATION)

        # Centers the window on the screen
        self.window.geometry('%dx%d+%d+%d' % (
            self.WINDOW_WIDTH_CLASSIFICATION, self.WIDOW_HEIGHT_CLASSIFICATION,
            (self.window.winfo_screenwidth() - self.WINDOW_WIDTH_CLASSIFICATION) / 2,
            (self.window.winfo_screenheight() - self.WIDOW_HEIGHT_CLASSIFICATION) / 2))

        # Defines font type, font size of all ttk widgets
        app_style = ttk.Style()

        app_style.configure('.', font=('Calibri', 12))

        style_white = ttk.Style()
        style_white.configure('wht.TFrame', background='#fff')

        self.entered_filepath = tk.StringVar()

        self.entered_w_1 = tk.StringVar(master=self.window, value=str(self.main_object.default_WEIGHT_1))
        self.entered_w_2 = tk.StringVar(master=self.window, value=str(self.main_object.default_WEIGHT_2))
        self.entered_threshold = tk.StringVar(master=self.window, value=str(self.main_object.default_THRESHOLD))
        self.entered_learning_rate = tk.StringVar(master=self.window, value=str(
            self.main_object.default_LEARNING_RATE_LINEAR_CLASSIFICATION))

        self.entered_epochs = tk.StringVar(master=self.window, value=str(self.main_object.default_EPOCHS))

        self.use_learning_rate = tk.IntVar()

        self.show_half_area = tk.IntVar()
        self.show_next_point_to_train = tk.IntVar()

        self.classification_point_enabled = tk.IntVar()
        self.entered_classify_input_x_1 = tk.StringVar(
            value=str(self.main_object.default_CLASSIFICATION_POINT_INPUT_X1))
        self.entered_classify_input_x_2 = tk.StringVar(
            value=str(self.main_object.default_CLASSIFICATION_POINT_INPUT_X2))
        self.entered_classify_target = tk.StringVar(value=str(self.main_object.default_CLASSIFICATION_POINT_TARGET))

        self.selected_mode = tk.StringVar()

        self.selected_display_mode_for_linear_regression = tk.StringVar()

        self.selected_display_mode_for_linear_classification = tk.StringVar()

        self.current_training_steps = tk.IntVar(value=0)

        self.frm_choose_mode = tk.LabelFrame(master=self.window, text="Modus ausw\u00E4hlen",
                                             font='Calibri 12 bold')
        self.frm_choose_mode.grid(column=0, row=0, sticky="ew", padx=(5, 5))
        self.com_box_mode_selection = ttk.Combobox(master=self.frm_choose_mode, state="readonly", width=24,
                                                   textvariable=self.selected_mode,
                                                   values=[self.mode_LINEAR_CLASSIFICATION,
                                                           self.mode_LINEAR_REGRESSION])
        self.com_box_mode_selection.current(0)
        self.selected_mode.trace_add(mode='write', callback=self.main_object.call_general_mode_switch)
        self.com_box_mode_selection.pack(side=tk.LEFT, padx=(2, 10), pady=6)

        # Inits and arranges the feature of choosing a train data set and display the selected filepath
        self.frm_choose_train_data = tk.LabelFrame(master=self.window, text="W\u00E4hlen Sie die Trainingsdaten aus:",
                                                   font='Calibri 12 bold', borderwidth=0)
        self.frm_choose_train_data.grid(column=1, row=0, columnspan=2, sticky="ew", padx=(5, 0))

        self.btn_choose_train_data = ttk.Button(master=self.frm_choose_train_data, text="\u00D6ffnen", width=15,
                                                command=lambda: self.main_object.open_file(
                                                    self.lbl_diplay_selected_filepath))
        self.btn_choose_train_data.pack(side=tk.LEFT, fill='x', expand=True)

        self.lbl_diplay_selected_filepath = ttk.Label(master=self.frm_choose_train_data, width=80)
        self.lbl_diplay_selected_filepath.pack(side=tk.LEFT, fill='x', expand=True)

        self.lbl_logo_1 = tk.Label(master=self.frm_choose_train_data, image=self.img_logo_university_1)
        self.lbl_logo_1.image = self.img_logo_university_1
        self.lbl_logo_1.pack(side=tk.LEFT, padx=(0, 5), fill='x', expand=True)

        # Inits and arranges the inputs of the parameters
        self.frm_entries = tk.LabelFrame(master=self.window, text="Geben Sie hier die Parameter ein:",
                                         font='Calibri 12 bold', borderwidth=0)
        self.frm_entries.grid(column=0, row=1, columnspan=3, sticky="ew", padx=(5, 0), pady=5)

        self.lbl_w_1 = ttk.Label(master=self.frm_entries, text="Gewicht w\u2081: ")
        self.lbl_w_1.pack(side=tk.LEFT)
        self.etr_w_1 = ttk.Entry(master=self.frm_entries, width=10, textvariable=self.entered_w_1)
        self.etr_w_1.pack(side=tk.LEFT)
        self.lbl_w_2 = ttk.Label(master=self.frm_entries, text="  Gewicht w\u2082: ")
        self.lbl_w_2.pack(side=tk.LEFT)
        self.etr_w_2 = ttk.Entry(master=self.frm_entries, width=10,
                                 textvariable=self.entered_w_2)
        self.etr_w_2.pack(side=tk.LEFT)

        self.lbl_threshold = ttk.Label(master=self.frm_entries, text="  Schwellenwert \u03B8: ")
        self.lbl_threshold.pack(side=tk.LEFT)
        self.etr_threshold = ttk.Entry(master=self.frm_entries, width=10, textvariable=self.entered_threshold)
        self.etr_threshold.pack(side=tk.LEFT)

        self.lbl_learning_rate = ttk.Label(master=self.frm_entries, text="  Lernrate \u03B1: ")
        self.etr_learning_rate = ttk.Entry(master=self.frm_entries, width=10, textvariable=self.entered_learning_rate)
        # widget for learning rate are shown/packed as soon as the according check button is clicked

        self.btn_initialize = ttk.Button(master=self.frm_entries, text="Initialisieren", width=15,
                                         command=self.main_object.call_initialize_perceptron)
        self.btn_initialize.pack(side=tk.LEFT, padx=5)

        self.ck_btn_learning_rate = ttk.Checkbutton(master=self.frm_entries, text="Lernrate einblenden",
                                                    style='TRadiobutton', takefocus=0,
                                                    variable=self.use_learning_rate, onvalue=1, offvalue=0,
                                                    command=self.main_object.call_switch_learning_rate_usage)
        self.ck_btn_learning_rate.pack(side=tk.RIGHT, padx=(0, 8))

        # Inits and arranges the visualization of the perceptron, training data and linear classifier(line)
        self.frm_display_visualization = ttk.Frame(master=self.window)
        self.frm_display_visualization.grid(column=0, row=2, columnspan=3, sticky="ew", padx=(1, 0))
        self.frm_display_perceptron = ttk.Frame(master=self.frm_display_visualization, relief=tk.GROOVE, borderwidth=5)
        self.frm_display_perceptron.grid(column=0, row=0, sticky="ns")

        # Necessary to make the sticky parameter work for the grid layout in Frame frm_display_perceptron
        self.frm_display_perceptron.grid_columnconfigure(0, weight=1)
        # Only the row that should expand as much as possible
        self.frm_display_perceptron.grid_rowconfigure(4, weight=1)

        self.canvas_perceptron = tk.Canvas(master=self.frm_display_perceptron, width=478, height=285,
                                           background="white")
        self.canvas_perceptron.grid(column=0, row=0, sticky="ew")
        self.canvas_img_id = self.canvas_perceptron.create_image(0, 0, image=self.img_perceptron_linear_classification,
                                                                 anchor="nw")
        self.canvas_text_id_w_1 = self.canvas_perceptron.create_text(106, 76, text="w\u2081",
                                                                     font='Calibri 14 bold', fill="blue")
        self.canvas_perceptron.itemconfig(self.canvas_text_id_w_1, angle=339)
        self.canvas_text_id_w_2 = self.canvas_perceptron.create_text(100, 145, text="w\u2082",
                                                                     font='Calibri 14 bold', fill="blue")
        self.canvas_perceptron.itemconfig(self.canvas_text_id_w_2, angle=20)
        self.canvas_text_id_threshold = self.canvas_perceptron.create_text(225, 237,
                                                                           text="\u03B8",
                                                                           font='Calibri 14 bold', fill="brown")

        self.canvas_text_id_input_x_1 = self.canvas_perceptron.create_text(36, 65, text="x\u2081",
                                                                           font='Calibri 18 bold', fill="black")
        self.canvas_text_id_input_x_2 = self.canvas_perceptron.create_text(36, 180,
                                                                           text="x\u2082",
                                                                           font='Calibri 18 bold', fill="black")

        self.canvas_arrow_id_w_1 = self.canvas_perceptron.create_line(56, 72, 150, 109, arrow=tk.LAST, width=7,
                                                                      fill="blue")
        self.canvas_arrow_id_w_2 = self.canvas_perceptron.create_line(56, 176, 150, 143, arrow=tk.LAST, width=7,
                                                                      fill="blue")

        self.canvas_text_id_calculated_output = self.canvas_perceptron.create_text(360, 134, text="Ausgabe",
                                                                                   font='Calibri 20 bold',
                                                                                   fill="purple", anchor=tk.W)

        self.canvas_text_id_expected_output = self.canvas_perceptron.create_text(475, 272, text="",
                                                                                 font='Calibri 18 bold',
                                                                                 fill="purple", anchor=tk.E)

        self.canvas_text_id_learining_rate = self.canvas_perceptron.create_text(6, 275, text="",
                                                                                font='Calibri 12',
                                                                                fill="black", anchor=tk.W)

        # Init arrows for later
        self.canvas_arrow_id_explanation_calculation_weighted_sum = self.canvas_perceptron.create_line(0, 0, 0, 0,
                                                                                                       arrow=tk.LAST,
                                                                                                       width=4,
                                                                                                       fill="green",
                                                                                                       dash=(20, 5))

        self.canvas_arrow_id_explanation_calculation_activation_function = self.canvas_perceptron.create_line(0, 0,
                                                                                                              0, 0,
                                                                                                              arrow=tk.LAST,
                                                                                                              width=4,
                                                                                                              fill="green",
                                                                                                              dash=(
                                                                                                                  20,
                                                                                                                  5))

        self.canvas_text_id_explanation_calculation_weighted_sum = self.canvas_perceptron.create_text(195, 23, text="",
                                                                                                      font='Calibri 12',
                                                                                                      fill="green")
        self.canvas_text_id_explanation_calculation_activation_function = self.canvas_perceptron.create_text(355, 48,
                                                                                                             text="",
                                                                                                             font='Calibri 12',
                                                                                                             fill="green",
                                                                                                             justify=tk.CENTER)

        # Frame for control buttons of the perceptron
        self.frm_perceptron_control = ttk.Frame(master=self.frm_display_perceptron)
        self.frm_perceptron_control.grid(column=0, row=1, sticky='ew')
        self.btn_train_auto_mode = ttk.Button(master=self.frm_perceptron_control,
                                              text="Automatisch trainieren",
                                              command=self.main_object.call_train_automode)
        self.btn_train_auto_mode.pack(side=tk.LEFT, fill='x', expand=True)
        self.btn_train_perceptron = ttk.Button(master=self.frm_perceptron_control, text="         Trainieren         ",
                                               command=self.main_object.call_train)
        self.btn_train_perceptron.pack(side=tk.LEFT, fill='x', expand=True)
        self.btn_train_reset = ttk.Button(master=self.frm_perceptron_control, text="      Zur\u00FCcksetzen      ",
                                          command=self.main_object.call_reset_perceptron)
        self.btn_train_reset.pack(side=tk.LEFT, fill='x', expand=True)

        # Init für das automatische Trainieren
        self.btn_train_automode_faster = ttk.Button(master=self.frm_perceptron_control,
                                                    text="Schneller",
                                                    command=self.main_object.call_auto_mode_running_speed_faster)
        self.btn_train_automode_slower = ttk.Button(master=self.frm_perceptron_control,
                                                    text="Langsamer",
                                                    command=self.main_object.call_auto_mode_running_speed_slower)

        # Init for later
        self.lbl_epochs = ttk.Label(master=self.frm_perceptron_control, text=" Epochen: ")
        self.etr_epochs = ttk.Entry(master=self.frm_perceptron_control, width=10, textvariable=self.entered_epochs)
        self.lbl_spacing = ttk.Label(master=self.frm_perceptron_control, text=" ", font='Calibri 19 ', width=1)

        self.frm_progress_classification = ttk.Frame(master=self.frm_display_perceptron, relief=tk.GROOVE,
                                                     borderwidth=5)
        self.frm_progress_classification.grid(column=0, row=2, sticky='ew')
        self.lbl_progress_classification = ttk.Label(master=self.frm_progress_classification,
                                                     text="Korrekt klassifiziert:  ",
                                                     font='Calibri 12 bold')
        self.lbl_progress_classification.pack(side=tk.LEFT)
        self.progressbar_classified_correctly = ttk.Progressbar(
            master=self.frm_progress_classification, mode='determinate')
        self.progressbar_classified_correctly.pack(side=tk.LEFT, fill='x', expand=True)

        self.frm_number_training_steps = ttk.Frame(master=self.frm_display_perceptron, relief=tk.GROOVE,
                                                   borderwidth=5)
        self.frm_number_training_steps.grid(column=0, row=3, sticky='new')
        self.lbl_num_training_step_desc = ttk.Label(master=self.frm_number_training_steps,
                                                    text="Bisherige Anzahl an Trainingsschritten: ",
                                                    font='Calibri 12 bold')
        self.lbl_num_training_step_desc.pack(side=tk.LEFT)
        self.lbl_num_training_steps = ttk.Label(master=self.frm_number_training_steps,
                                                textvariable=self.current_training_steps, font='Calibri 12')
        self.lbl_num_training_steps.pack(side=tk.LEFT)

        # Frame for classification of a new data point
        self.frm_classify_point = ttk.Frame(master=self.frm_display_perceptron, relief=tk.GROOVE, borderwidth=5)
        self.frm_classify_point.grid(column=0, row=4, sticky="nsew")

        self.ck_btn_classifiy_point = ttk.Checkbutton(master=self.frm_classify_point, text="Neuen Punkt klassifizieren",
                                                      variable=self.classification_point_enabled, onvalue=1, offvalue=0,
                                                      style='TRadiobutton', takefocus=0,
                                                      command=self.main_object.call_display_classification)
        self.ck_btn_classifiy_point.grid(column=0, row=0, sticky='w')
        self.frm_classify_inputs = tk.LabelFrame(master=self.frm_classify_point, text="Eingaben",
                                                 font='Calibri 12 bold')
        self.lbl_classify_input_x_1 = ttk.Label(master=self.frm_classify_inputs, text=' x\u2081:')
        self.etr_classify_input_x_1 = ttk.Entry(master=self.frm_classify_inputs,
                                                textvariable=self.entered_classify_input_x_1,
                                                width=18)
        self.lbl_classify_input_x_2 = ttk.Label(master=self.frm_classify_inputs, text=' x\u2082:')
        self.etr_classify_input_x_2 = ttk.Entry(master=self.frm_classify_inputs,
                                                textvariable=self.entered_classify_input_x_2, width=18)
        self.frm_classify_target = tk.LabelFrame(master=self.frm_classify_point, text="Erwartete Ausgabe",
                                                 font='Calibri 12 bold')
        self.lbl_classify_target = ttk.Label(master=self.frm_classify_target, text=' Ausgabe:')
        self.etr_classify_target = ttk.Entry(master=self.frm_classify_target,
                                             textvariable=self.entered_classify_target, width=18)
        self.btn_classify_point = ttk.Button(master=self.frm_classify_point,
                                             text="Klassifizieren(Phase 1: Berechne Perzeptron f\u00FCr Eingabepunkt)",
                                             command=self.main_object.call_classification)

        self.lbl_classify_input_x_1.pack(side=tk.LEFT)
        self.etr_classify_input_x_1.pack(side=tk.LEFT, fill='x')
        self.lbl_classify_input_x_2.pack(side=tk.LEFT)
        self.etr_classify_input_x_2.pack(side=tk.LEFT, fill='x', padx=2)
        self.lbl_classify_target.pack(side=tk.LEFT)
        self.etr_classify_target.pack(side=tk.LEFT, fill='x')

        # Frames for visualizing training data
        self.frm_display_plotting = ttk.Frame(master=self.frm_display_visualization, relief=tk.GROOVE, borderwidth=5)
        self.frm_display_plotting.grid(column=1, row=0)

        # Creates a figure with 6*100 x 4*100 pixels
        self.figure_training_data = Figure(figsize=(6, 4.5), dpi=100)

        # Attribute for the displayed plotting ==> always use for displaying plotting updates,...
        self.plot2D = self.figure_training_data.add_subplot(111)
        self.plot2D.set_xlabel("x\u2081", fontsize=14)
        self.plot2D.set_ylabel("x\u2082", fontsize=14)
        self.canvas = FigureCanvasTkAgg(self.figure_training_data, master=self.frm_display_plotting)
        self.canvas.get_tk_widget().grid(column=0, row=0, columnspan=3)

        self.figure_training_data.clear()
        self.plot3D = self.figure_training_data.add_subplot(111, projection='3d')
        self.plot3D.set_box_aspect(aspect=(1, 1, 1))

        self.figure_training_data.clear()
        self.figure_training_data.add_axes(self.plot2D)

        # delete unwanted features
        NavigationToolbar2Tk.toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                                          t[0] not in ('Pan', 'Subplots', 'Forward')]
        # setting up toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frm_display_plotting, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(column=0, row=1, sticky='nsew', pady=(4, 4))

        self.frm_radio_buttons_half_area_next_point_to_train = ttk.Frame(master=self.frm_display_plotting)
        self.frm_display_plotting.columnconfigure(1, weight=3)
        self.frm_radio_buttons_half_area_next_point_to_train.grid(column=1, row=1, sticky='e', padx=2)

        self.ck_btn_show_half_areas = ttk.Checkbutton(master=self.frm_radio_buttons_half_area_next_point_to_train,
                                                      text="Halbr\u00E4ume anzeigen",
                                                      variable=self.show_half_area, onvalue=1, offvalue=0,
                                                      style='TRadiobutton', takefocus=0,
                                                      command=self.main_object.call_replot_show_half_areas)
        self.ck_btn_show_half_areas.pack(anchor='w')
        self.ck_btn_show_next_point_to_train = ttk.Checkbutton(
            master=self.frm_radio_buttons_half_area_next_point_to_train,
            text="N\u00E4chster Trainingspunkt",
            variable=self.show_next_point_to_train, onvalue=1, offvalue=0,
            style='TRadiobutton', takefocus=0,
            command=self.main_object.call_replot_show_next_point_to_train)
        self.ck_btn_show_next_point_to_train.pack(anchor='w')
        self.frm_choose_display_mode = tk.LabelFrame(master=self.frm_display_plotting, text="Ansicht",
                                                     font='Calibri 12 bold', borderwidth=0)
        self.frm_choose_display_mode.grid(column=2, row=1, sticky='n')
        self.com_box_lin_classification_display_mode = ttk.Combobox(master=self.frm_choose_display_mode,
                                                                    state="readonly",
                                                                    textvariable=self.selected_display_mode_for_linear_classification,
                                                                    values=[
                                                                        self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE,
                                                                        self.mode_LINEAR_CLASSIFICATION_WEIGHT_UPDATES])
        self.com_box_lin_classification_display_mode.current(0)
        self.com_box_lin_classification_display_mode.pack(side=tk.LEFT)
        self.selected_display_mode_for_linear_classification.trace_add(mode='write',
                                                                       callback=self.main_object.call_display_mode_switched_linear_classification)

        # preparation for linear regression
        self.com_box_lin_regression_display_mode = ttk.Combobox(master=self.frm_choose_display_mode, state="readonly",
                                                                textvariable=self.selected_display_mode_for_linear_regression,
                                                                values=[self.mode_LINEAR_REGRESSION_REGRESSION_LINE,
                                                                        self.mode_LINEAR_REGRESSION_GRADIENT_DESCENT])
        self.com_box_lin_regression_display_mode.current(0)
        self.selected_display_mode_for_linear_regression.trace_add(mode='write',
                                                                   callback=self.main_object.call_display_mode_switched_linear_regression)

        # Anlegen der Darstellungsmöglichkeit für die Anzeige Erklärungen Lineare Klassifikation
        self.frm_explanation_text_classification = ttk.Frame(master=self.window, relief=tk.GROOVE, borderwidth=5,
                                                             style='wht.TFrame')
        self.frm_explanation_text_classification.grid(column=0, row=3, columnspan=3, sticky="ew", padx=(1, 1))

        self.frm_train_protocol = tk.LabelFrame(master=self.frm_explanation_text_classification,
                                                text="Trainingsprotokoll", background="white",
                                                font='Calibri 12 bold', borderwidth=1, relief=tk.GROOVE)
        self.frm_train_protocol.grid(column=0, row=0, padx=(1, 1))

        self.lbl_explanation_training_protocol = tk.Label(master=self.frm_train_protocol, font=("Calibri", 12),
                                                          background='white', text="", width=59,
                                                          height=7, anchor=tk.NW, justify=tk.LEFT)
        self.lbl_explanation_training_protocol.pack(side=tk.LEFT, expand=True, fill='x')
        self.frm_separation_line = tk.LabelFrame(master=self.frm_explanation_text_classification,
                                                 text="Trenngerade", background="white",
                                                 font='Calibri 12 bold', borderwidth=1, relief=tk.GROOVE)
        self.frm_separation_line.grid(column=1, row=0, padx=(5, 1))

        self.lbl_explanation_separation_line = tk.Label(master=self.frm_separation_line, font=("Calibri", 12),
                                                        background='white', width=75,
                                                        height=7, anchor=tk.NW, justify=tk.LEFT)
        self.lbl_explanation_separation_line.pack(side=tk.LEFT, expand=True, fill='x')

        # Anlegen der Darstellungsmöglichkeit für die Anzeige der Gleichung der Regressionsgerade
        self.frm_explanation_text_regression = ttk.Frame(master=self.window, relief=tk.GROOVE, borderwidth=5,
                                                         style='wht.TFrame')

        self.lbl_explanation_regression = tk.Label(master=self.frm_explanation_text_regression, font=("Calibri", 12),
                                                   background='white',
                                                   height=2, anchor=tk.NW, justify=tk.LEFT)
        self.lbl_explanation_regression.pack(side=tk.LEFT, expand=True, fill='x')

        # Anlegen der Credit-Zeile
        self.frm_credits = ttk.Frame(master=self.window)
        self.frm_credits.grid(column=0, row=4, columnspan=3, sticky="ew")
        self.lbl_didaktik = tk.Label(master=self.frm_credits, anchor=tk.E, justify=tk.LEFT,
                                     text=" Didaktik der Informatik - Universit\u00E4t Passau",
                                     font=("Calibri Italic", 11))
        self.lbl_didaktik.pack(side=tk.LEFT)
        self.lbl_university = tk.Label(master=self.frm_credits, anchor=tk.E, justify=tk.LEFT,
                                       text="Tobias Fuchs, Wolfgang Pfeffer  ", font=("Calibri Italic", 11))
        self.lbl_university.pack(side=tk.RIGHT)

    def adapt_gui_for_mode(self):
        """
        Passt die Oberfläche für den in der GUI ausgewählten Modus(Lineare Klassifiaktion oder Lineare Regression) an
        Setzt außerdem dem den angezeigten Dateipfad der Trainingsdaten zurück
        """
        self.entered_filepath.set("")
        self.lbl_diplay_selected_filepath['text'] = " "

        # Auswahlmöglichkeit des Modus aktivieren
        self.adapt_enable_disable_mode_selection(enable_mode_selection=True)

        # Alle Eingabemöglichkeiten aktivieren
        self.adapt_enable_disable_input_controls(enable_inputs=True)

        self.adapt_perceptron_control_buttons(self.mode_LINEAR_CLASSIFICATION, self.classification_point_enabled.get())

        self.adapt_plot_contol_options(self.mode_LINEAR_CLASSIFICATION, self.classification_point_enabled.get(),
                                       self.selected_display_mode_for_linear_classification.get())

        if self.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            self.adapt_gui_linear_classification()

        elif self.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            self.adapt_gui_linear_regression()

    def adapt_gui_linear_regression(self):
        """
        Passt die Oberfläche für den Modus "Lineare Regression" an
        Anzeige und Ausblenden von Buttons, Eingabefeldern,...
        Erzeugt die Initial-Oberfläche
        """
        # Adapt param entries
        self.lbl_w_2.pack_forget()
        self.etr_w_2.pack_forget()
        self.lbl_learning_rate.pack_forget()
        self.etr_learning_rate.pack_forget()
        self.lbl_threshold['text'] = "  Gewicht w\u2080: "

        self.adapt_gui_for_learning_rate()

        self.btn_train_perceptron.pack_forget()
        self.btn_train_reset.pack_forget()

        self.adapt_perceptron_control_buttons(self.mode_LINEAR_REGRESSION, self.classification_point_enabled.get())
        self.frm_progress_classification.grid_forget()

        # Blendet die Klassifizierungsmöglichkeit eines Punkts aus
        # Wert der Variable der Checkbox wird in Main gesetzt
        self.ck_btn_classifiy_point.grid_forget()
        self.adapt_classification_of_a_point_control(current_mode=self.selected_mode.get(),
                                                     display_mode=self.selected_display_mode_for_linear_regression.get(),
                                                     classification_of_a_point_enabled=self.classification_point_enabled.get(),
                                                     phase_of_classification="phase_0")

        self.adapt_plot_contol_options(self.mode_LINEAR_REGRESSION, self.classification_point_enabled.get(),
                                       self.selected_display_mode_for_linear_regression.get())

        # Anzeige des Erklärungsbereichs
        self.frm_explanation_text_classification.grid_forget()
        self.frm_explanation_text_regression.grid(column=0, row=3, columnspan=3, sticky="ew", padx=(1, 1))

        # set minimum window size value
        self.window.minsize(self.WINDOW_WIDTH_REGRESSION, self.WIDOW_HEIGHT_REGRESSION)

        # set maximum window size value
        self.window.maxsize(self.WINDOW_WIDTH_REGRESSION, self.WIDOW_HEIGHT_REGRESSION)

    def adapt_gui_linear_classification(self):
        """
        Passt die Oberfläche für den Modus "Lineare Klassifikation" an
        Anzeige und Ausblenden von Buttons, Eingabefeldern,...
        Erzeugt die Initial-Oberfläche
        """
        # Adapt param entries
        self.lbl_threshold.pack_forget()
        self.etr_threshold.pack_forget()
        self.lbl_learning_rate.pack_forget()
        self.etr_learning_rate.pack_forget()
        self.btn_initialize.pack_forget()

        self.lbl_w_2.pack(side=tk.LEFT)
        self.etr_w_2.pack(side=tk.LEFT)
        self.lbl_threshold.pack(side=tk.LEFT)
        self.lbl_threshold['text'] = "  Schwellenwert \u03B8: "
        self.etr_threshold.pack(side=tk.LEFT)
        self.btn_initialize.pack(side=tk.LEFT, padx=5)

        self.adapt_gui_for_learning_rate()

        self.adapt_perceptron_control_buttons(self.mode_LINEAR_CLASSIFICATION, self.classification_point_enabled.get())
        self.frm_progress_classification.grid(column=0, row=2, sticky='ew')

        self.ck_btn_classifiy_point.grid(column=0, row=0, sticky='w')
        self.adapt_classification_of_a_point_control(current_mode=self.selected_mode.get(),
                                                     display_mode=self.selected_display_mode_for_linear_classification.get(),
                                                     classification_of_a_point_enabled=self.classification_point_enabled.get(),
                                                     phase_of_classification="phase_0")

        self.adapt_plot_contol_options(self.mode_LINEAR_CLASSIFICATION, self.classification_point_enabled.get(),
                                       self.selected_display_mode_for_linear_classification.get())

        # Anzeige des Erklärungsbereichs
        self.frm_explanation_text_regression.grid_forget()
        self.frm_explanation_text_classification.grid(column=0, row=3, columnspan=3, sticky="ew", padx=(1, 1))

        # set minimum window size value
        self.window.minsize(self.WINDOW_WIDTH_CLASSIFICATION, self.WIDOW_HEIGHT_CLASSIFICATION)

        # set maximum window size value
        self.window.maxsize(self.WINDOW_WIDTH_CLASSIFICATION, self.WIDOW_HEIGHT_CLASSIFICATION)

    def adapt_classification_of_a_point_control(self, current_mode, display_mode, classification_of_a_point_enabled=0,
                                                phase_of_classification="phase_0"):
        if current_mode == self.mode_LINEAR_CLASSIFICATION and display_mode == self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE:
            self.frm_classify_point.grid(column=0, row=4, sticky="nsew")
            self.ck_btn_classifiy_point.grid(column=0, row=0, sticky='w')
            if classification_of_a_point_enabled == 0:
                # Ausblenden der Oberfläche für die Klassifikation eines Punkts
                self.frm_classify_inputs.grid_forget()
                self.frm_classify_target.grid_forget()
                self.btn_classify_point.grid_forget()
            elif classification_of_a_point_enabled == 1:
                # Anzeige der Oberfläche für die Klassifikation eines Punkts

                self.frm_classify_inputs.grid(column=0, row=1, sticky='sew')
                self.frm_classify_target.grid(column=1, row=1, sticky='sew')
                # Necessary to make the sticky parameter work for the grid layout in Frame frm_display_perceptron
                self.frm_classify_point.grid_columnconfigure(0, weight=1)
                self.frm_classify_point.grid_rowconfigure(2, weight=1)

                self.btn_classify_point.grid(column=0, row=2, columnspan=2, sticky='ews')

                if phase_of_classification == "phase_2":
                    # Herstellen von Phase 2
                    self.etr_classify_input_x_1['state'] = tk.DISABLED
                    self.etr_classify_input_x_2['state'] = tk.DISABLED
                    self.etr_classify_target['state'] = tk.DISABLED
                    self.btn_classify_point[
                        'text'] = "Klassifizieren (Phase 1: Berechne Perzeptron für Eingabepunkt)"

                elif phase_of_classification == "phase_3":
                    # Herstellen von Phase 3
                    self.etr_classify_input_x_1['state'] = tk.DISABLED
                    self.etr_classify_input_x_2['state'] = tk.DISABLED
                    self.etr_classify_target['state'] = tk.DISABLED
                    self.btn_classify_point['text'] = "Klassifizieren (Phase 2: Vergleich berechnet - erwartet)"

                elif phase_of_classification == "phase_1":
                    # Herstellen von Phase 1
                    self.etr_classify_input_x_1['state'] = tk.NORMAL
                    self.etr_classify_input_x_2['state'] = tk.NORMAL
                    self.etr_classify_target['state'] = tk.NORMAL
                    self.btn_classify_point['text'] = "Punkt anzeigen"
                elif phase_of_classification == "phase_0":
                    # Herstellen von Phase 0
                    self.frm_classify_inputs.grid_forget()
                    self.frm_classify_target.grid_forget()
                    self.btn_classify_point['text'] = "Neue Klassifikation starten"

        else:
            self.frm_classify_point.grid_forget()

    # NICHT VERWENDET
    def adapt_gui_for_classification_of_a_point(self):
        classification_enabled = self.classification_point_enabled.get()

        if classification_enabled == 1:
            # Alle Eingabemöglichkeiten sperren
            self.btn_choose_train_data['state'] = tk.DISABLED
            self.etr_w_1['state'] = tk.DISABLED
            self.etr_w_2['state'] = tk.DISABLED
            self.etr_threshold['state'] = tk.DISABLED
            self.btn_initialize['state'] = tk.DISABLED
            self.ck_btn_learning_rate['state'] = tk.DISABLED

            self.adapt_perceptron_control_buttons(self.mode_LINEAR_CLASSIFICATION, classification_enabled)

            self.adapt_plot_contol_options(self.mode_LINEAR_CLASSIFICATION, classification_enabled,
                                           self.selected_display_mode_for_linear_classification.get())

            # Anzeige der Oberfläche für die Klassifikation eines Punkts
            # setzt die Eingabefelder auf Standardwerte
            self.entered_classify_input_x_1.set(str(1))
            self.entered_classify_input_x_2.set(str(1))
            self.entered_classify_target.set(str(1))

            self.frm_classify_inputs.grid(column=0, row=1, sticky='ew')
            self.lbl_classify_input_x_1.pack(side=tk.LEFT)
            self.etr_classify_input_x_1.pack(side=tk.LEFT, fill='x')
            self.lbl_classify_input_x_2.pack(side=tk.LEFT)
            self.etr_classify_input_x_2.pack(side=tk.LEFT, fill='x', padx=2)
            self.frm_classify_target.grid(column=1, row=1, sticky='ew')
            self.lbl_classify_target.pack(side=tk.LEFT)
            self.etr_classify_target.pack(side=tk.LEFT, fill='x')
            self.btn_classify_point.grid(column=0, row=2, columnspan=2, sticky='ew')

            self.etr_classify_input_x_1['state'] = tk.NORMAL
            self.etr_classify_input_x_2['state'] = tk.NORMAL
            self.etr_classify_target['state'] = tk.NORMAL
            self.btn_classify_point['text'] = "Punkt anzeigen"

        else:
            # Alle Eingabemöglichkeiten aktivieren
            self.btn_choose_train_data['state'] = tk.NORMAL
            self.etr_w_1['state'] = tk.NORMAL
            self.etr_w_2['state'] = tk.NORMAL
            self.etr_threshold['state'] = tk.NORMAL
            self.btn_initialize['state'] = tk.NORMAL
            self.ck_btn_learning_rate['state'] = tk.NORMAL

            self.adapt_perceptron_control_buttons(self.mode_LINEAR_CLASSIFICATION, classification_enabled)

            self.adapt_plot_contol_options(self.mode_LINEAR_CLASSIFICATION, classification_enabled,
                                           self.selected_display_mode_for_linear_classification.get())

            # Ausblenden der Oberfläche für die Klassifikation eines Punkts
            self.frm_classify_inputs.grid_forget()
            self.lbl_classify_input_x_1.pack_forget()
            self.etr_classify_input_x_1.pack_forget()
            self.lbl_classify_input_x_2.pack_forget()
            self.etr_classify_input_x_2.pack_forget()
            self.frm_classify_target.grid_forget()
            self.lbl_classify_target.pack_forget()
            self.etr_classify_target.pack_forget()
            self.btn_classify_point.grid_forget()

    def adapt_gui_for_learning_rate(self):
        """
        Passt die Oberfläche für die Eingabe der Lernrate an. Falls die Checkbox ausgewählt, wird
        eine Eingabemöglichkeit angezeigt. Falls nicht, wird die Eingabemöglichkeit ausgeblendet
        """
        if self.use_learning_rate.get() == 1:
            self.btn_initialize.pack_forget()
            self.lbl_learning_rate.pack(side=tk.LEFT)
            self.etr_learning_rate.pack(side=tk.LEFT)
            self.btn_initialize.pack(side=tk.LEFT, padx=5)
        else:
            self.lbl_learning_rate.pack_forget()
            self.etr_learning_rate.pack_forget()

    # NICHT VERWENDET
    def adapt_gui_for_separation_line_OR_gradient_descent_lin_classification(self):
        if self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE:
            self.btn_train_auto_mode.pack_forget()
            self.btn_train_perceptron.pack_forget()
            self.btn_train_reset.pack_forget()
            self.btn_train_auto_mode.pack(side=tk.LEFT, fill='x', expand=True)
            self.btn_train_perceptron.pack(side=tk.LEFT, fill='x', expand=True)
            self.btn_train_reset.pack(side=tk.LEFT, fill='x', expand=True)
            self.frm_radio_buttons_half_area_next_point_to_train.grid(column=1, row=1, sticky='e', padx=2)
            self.ck_btn_classifiy_point.grid(column=0, row=0, sticky='w')
        elif self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_WEIGHT_UPDATES:
            self.btn_train_auto_mode.pack_forget()
            self.frm_radio_buttons_half_area_next_point_to_train.grid_forget()
            self.btn_train_perceptron.pack_configure(fill=tk.X, expand=True)
            self.btn_train_reset.pack_configure(fill=tk.X, expand=True)
            self.ck_btn_classifiy_point.grid_forget()

    def adapt_enable_disable_mode_selection(self, enable_mode_selection):
        """
        Aktiviert oder deaktitiviert die Auswahlmöglichkeit des Modus
        :param enable_mode_selection: True, wenn die Kombo-Box für die Auswahl des Modus aktiviert werden soll
        """
        if enable_mode_selection:
            self.com_box_mode_selection['state'] = "readonly"
        else:
            self.com_box_mode_selection['state'] = tk.DISABLED

    def adapt_enable_disable_input_controls(self, enable_inputs):
        """
        Aktiviert oder sperrt die Eingabekontrollmöglichkeiten für Daten und Parameter des Perzeptrons
        :param enable_inputs: Aktivieren der Eingabemöglichkeiten, falls True
        """
        if enable_inputs:
            # Alle Eingabemöglichkeiten aktivieren
            self.btn_choose_train_data['state'] = tk.NORMAL
            self.etr_w_1['state'] = tk.NORMAL
            self.etr_w_2['state'] = tk.NORMAL
            self.etr_threshold['state'] = tk.NORMAL
            self.btn_initialize['state'] = tk.NORMAL
            self.ck_btn_learning_rate['state'] = tk.NORMAL
            self.etr_learning_rate['state'] = tk.NORMAL
        else:
            # Alle Eingabemöglichkeiten aktivieren
            self.btn_choose_train_data['state'] = tk.DISABLED
            self.etr_w_1['state'] = tk.DISABLED
            self.etr_w_2['state'] = tk.DISABLED
            self.etr_threshold['state'] = tk.DISABLED
            self.btn_initialize['state'] = tk.DISABLED
            self.ck_btn_learning_rate['state'] = tk.DISABLED
            self.etr_learning_rate['state'] = tk.DISABLED

    def adapt_plot_contol_options(self, current_mode, classification_of_a_point_enabled=0, display_mode="",
                                  auto_mode_running=False):
        """
        Regelt die korrekte Anzeige der Kontroll-Möglichkeiten des Plots in Abhängigkeit des gewählten Modus und der
        Anzeigeeinstellungen
        :param current_mode: ausgwählter Modus (Lineare Klassifikation, Lineare Regression)
        :param classification_of_a_point_enabled: 0 oder 1 Gibt an, ob momentan die Klassifikation eines Punkts aktiviert ist
        :param display_mode: Nur für Modus "Lineare Klassifikation" relevant. Wählbare Modi (Trenngerade, Gewichtsaktualisierung)
        """
        self.frm_choose_display_mode.grid(column=2, row=1, sticky='n')
        if current_mode == self.mode_LINEAR_REGRESSION:
            self.frm_radio_buttons_half_area_next_point_to_train.grid_forget()
            self.com_box_lin_classification_display_mode.pack_forget()
            self.com_box_lin_regression_display_mode.pack(side=tk.LEFT)
        elif current_mode == self.mode_LINEAR_CLASSIFICATION:
            self.com_box_lin_regression_display_mode.pack_forget()
            if display_mode == self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE:
                self.frm_radio_buttons_half_area_next_point_to_train.grid(column=1, row=1, sticky='e', padx=2)
                if classification_of_a_point_enabled == 1:
                    self.ck_btn_show_next_point_to_train.pack_forget()
                    self.frm_choose_display_mode.grid_forget()
                else:
                    self.ck_btn_show_next_point_to_train.pack(anchor='w')
                    self.com_box_lin_classification_display_mode.pack(side=tk.LEFT)

                if auto_mode_running:
                    self.toolbar.grid_forget()
                    self.com_box_lin_classification_display_mode['state'] = tk.DISABLED
                    # self.ck_btn_show_half_areas['state'] = tk.DISABLED
                    # self.ck_btn_show_next_point_to_train['state'] = tk.DISABLED

                else:
                    self.toolbar.grid(column=0, row=1, sticky='nsew', pady=(4, 4))
                    self.com_box_lin_classification_display_mode['state'] = "readonly"
                    # self.ck_btn_show_half_areas['state'] = tk.NORMAL
                    # self.ck_btn_show_next_point_to_train['state'] = tk.NORMAL

            elif display_mode == self.mode_LINEAR_CLASSIFICATION_WEIGHT_UPDATES:
                self.com_box_lin_classification_display_mode.pack(side=tk.LEFT)
                self.frm_radio_buttons_half_area_next_point_to_train.grid_forget()

    def adapt_perceptron_control_buttons(self, current_mode, classification_of_a_point_enabled=0,
                                         auto_mode_running=False, auto_mode_running_speed=1,
                                         display_mode="Trenngerade"):
        """
        Regelt die korrekte Anzeige der Kontroll-Buttons des Perzeptrons in Abhängigkeit des gewählten Modus und der
        Anzeigeeinstellungen
        :param display_mode: Nur für Modus "Lineare Klassifikation" relevant, für die Anzeige des auto-mode-Buttons nötig
        :param current_mode: ausgwählter Modus (Lineare Klassifikation, Lineare Regression)
        :param classification_of_a_point_enabled: 0 oder 1 Gibt an, ob momentan die Klassifikation eines Punkts aktiviert ist
        :param auto_mode_running: Nur im Modus "Lineare Klassifikation" von Bedeutung; Falls True werden die Kontrollbuttons
        für den Automodus angezeigt, bei False ausgeblendet
        :param auto_mode_running_speed: Nur im Modus "Lineare Klassifikation" von Bedeutung; Wird benötigt um ggf. den
        Schneller/Langsamer-Button zu deaktivieren, wenn die Geschwindigkeit zu hoch werden würde
        """
        if current_mode == self.mode_LINEAR_REGRESSION:
            self.btn_train_auto_mode.pack_forget()
            self.btn_train_automode_faster.pack_forget()
            self.btn_train_automode_slower.pack_forget()
            self.btn_train_perceptron.pack_forget()
            self.btn_train_reset.pack_forget()
            self.lbl_epochs.pack_forget()
            self.etr_epochs.pack_forget()
            self.lbl_spacing.pack_forget()

            self.btn_train_perceptron['state'] = tk.NORMAL
            self.btn_train_perceptron.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            self.lbl_epochs.pack(side=tk.LEFT)
            self.etr_epochs.pack(side=tk.LEFT)
            self.lbl_spacing.pack(side=tk.LEFT)
            self.btn_train_reset.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.btn_train_reset['state'] = tk.NORMAL

        elif current_mode == self.mode_LINEAR_CLASSIFICATION:
            self.btn_train_auto_mode.pack_forget()
            self.btn_train_automode_faster.pack_forget()
            self.btn_train_automode_slower.pack_forget()
            self.btn_train_perceptron.pack_forget()
            self.btn_train_reset.pack_forget()
            self.lbl_epochs.pack_forget()
            self.etr_epochs.pack_forget()
            self.lbl_spacing.pack_forget()

            if display_mode == self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE:
                self.btn_train_auto_mode.pack(side=tk.LEFT, fill='x', expand=True)

            if not auto_mode_running:
                self.btn_train_auto_mode['text'] = "Automatisch trainieren"
                self.btn_train_perceptron.pack(side=tk.LEFT, fill='x', expand=True)
                self.btn_train_reset.pack(side=tk.LEFT, fill='x', expand=True)
                if classification_of_a_point_enabled == 1:
                    self.btn_train_auto_mode['state'] = tk.DISABLED
                    self.btn_train_reset['state'] = tk.DISABLED
                    self.btn_train_perceptron['state'] = tk.DISABLED
                else:
                    self.btn_train_auto_mode['state'] = tk.NORMAL
                    self.btn_train_reset['state'] = tk.NORMAL
                    self.btn_train_perceptron['state'] = tk.NORMAL
            else:
                self.btn_train_automode_slower.pack(side=tk.LEFT, fill='x', expand=True)
                self.btn_train_automode_faster.pack(side=tk.LEFT, fill='x', expand=True)
                self.btn_train_auto_mode['text'] = "Training stoppen"

    def reset_plotting_figure(self, plotmode):
        """
        Reinigt die Plotting-Figur und setzt den Plot auf den leeren Plot des gewählten Modus mode
        !Achtung!: Ein Neuzeichnen der Fläche mit (...canvas.draw()) muss noch extra
        ausgeführt werden
        :param plotmode: Anzeigeeinstellung des neuen Plots - 2D oder 3D
        """
        self.figure_training_data.clear()
        if plotmode == "3D":
            self.figure_training_data.add_axes(self.plot2D)
            self.plot2D.cla()
        elif plotmode == "2D":
            self.figure_training_data.add_axes(self.plot2D)
            self.plot3D.cla()

    def clear_plot(self):
        """
        Löscht den aktuellen Plot.
        Dieser wird über den gewählten Modus und die Anzeigeeinstellung ausgewählt
        !Achtung!: Ein Neuzeichnen der Fläche mit (...canvas.draw()) muss noch extra
        ausgeführt werden
        """
        if self.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            if self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE:
                self.plot2D.cla()
            elif self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_WEIGHT_UPDATES:
                self.plot3D.cla()
        elif self.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            if self.selected_display_mode_for_linear_regression.get() == self.mode_LINEAR_REGRESSION_REGRESSION_LINE:
                self.plot2D.cla()
            elif self.selected_display_mode_for_linear_regression.get() == self.mode_LINEAR_REGRESSION_GRADIENT_DESCENT:
                self.plot3D.cla()

    # NICHT VERWENDET
    def plot_training_data(self, header_data, training_data):
        self.figure_training_data.clear()

        if self.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            if self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE:
                self.plot2D.cla()
            elif self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_WEIGHT_UPDATES:
                self.plot3D.cla()
        elif self.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            if self.selected_display_mode_for_linear_regression.get() == self.mode_LINEAR_REGRESSION_REGRESSION_LINE:
                self.plot2D.cla()
            elif self.selected_display_mode_for_linear_regression.get() == self.mode_LINEAR_REGRESSION_GRADIENT_DESCENT:
                self.plot3D.cla()

        self.figure_training_data.add_axes(self.plot2D)
        my_Plotting.plotting_data(self.plot2D, header_data,
                                  training_data, self.selected_mode.get(), self.main_object.plot_dimensions)
        my_Plotting.plotting_axis_arrows(self.plot2D)
        self.canvas.draw()

    def update_progressbar(self, number_of_correct, number_of_datasets):
        """
        Aktualisiert die Anzeige der Progressbar für die korrekt klassifizierten Datenobjekte
        :param number_of_correct: Anzahl der korrekt klassifizierten Trainingsdaten
        :param number_of_datasets: Anzahl der vorhandenen Trainingsdaten
        """
        self.progressbar_classified_correctly['value'] = number_of_correct * (100 / number_of_datasets)

    def update_perceptron_visualization(self, display_training_step, current_perceptron, temp_w_x=1, temp_w_y=1,
                                        temp_threshold=0, training_data_point=None, calculated_output=0,
                                        current_state="training", classification_first_phase_0=False):
        """
        Methode regelt die Anzeige des Perzeptrons für den aktuellen Modus. Der aktuelle Modus wird automatisch aus dem
        Attribut ausgelesen.
        :param display_training_step: (Nur für Lineare Klassifikation relevant) Anzeige der Anpassung der Gewichte
        :param current_perceptron: aktuelles Perzeptron; Falls None wird ein Blanko-Perzeptron angezeigt
        :param temp_w_x: (Nur für Lineare Klassifikation relevant) Gewicht w1 vor dem Trainingsschritt
        :param temp_w_y: (Nur für Lineare Klassifikation relevant) Gewicht w2 vor dem Trainingsschritt
        :param temp_threshold: (Nur für Lineare Klassifikation relevant) Schwellenwert vor dem Trainingsschritt
        :param training_data_point: (Nur für Lineare Klassifikation relevant) momentan trainierter Datenpunkt
        :param calculated_output: (Nur für Lineare Klassifikation relevant) für den Datenpunkt berechnete Ausgabe des
        Perzeptrons
        :param current_state: (Nur für Lineare Klassifikation relevant) "training": Klassifikation eines Punkts ist
        nicht aktiv, "classification": Klassifikation eines Punkts ist aktiv
        """
        if training_data_point is None:
            training_data_point = []

        # Ausblenden der Anzeige, welche bei der Klassifikation eines Punkts extra eingeblendet wird
        # self.canvas_perceptron.itemconfig(self.canvas_text_id_expected_output, text="", fill="purple")
        self.canvas_perceptron.coords(self.canvas_arrow_id_explanation_calculation_weighted_sum, 0, 0, 0, 0)
        self.canvas_perceptron.coords(self.canvas_arrow_id_explanation_calculation_activation_function, 0, 0, 0, 0)
        self.canvas_perceptron.itemconfig(self.canvas_text_id_explanation_calculation_weighted_sum, text="")

        self.canvas_perceptron.itemconfig(self.canvas_text_id_explanation_calculation_activation_function, text="")
        self.canvas_perceptron.itemconfig(self.canvas_text_id_expected_output, text="", fill="purple")

        if self.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:

            # Adapt displaying of perceptron
            self.canvas_perceptron.itemconfig(self.canvas_img_id, image=self.img_perceptron_linear_classification)

            if current_perceptron is None:
                self.canvas_perceptron.coords(self.canvas_arrow_id_w_1, 56, 72, 150, 109)
                self.canvas_perceptron.itemconfigure(self.canvas_arrow_id_w_1, width=7)

                self.canvas_perceptron.coords(self.canvas_arrow_id_w_2, 56, 176, 150, 143)
                self.canvas_perceptron.itemconfigure(self.canvas_arrow_id_w_2, width=7)

                self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_1,
                                                     text="w\u2081", fill="blue")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_2,
                                                     text="w\u2082", fill="blue")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_threshold,
                                                     text="\u03B8",
                                                     fill="brown")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_1, text="x\u2081")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2, text="x\u2082")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_calculated_output, text="Ausgabe")

                self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                     text="")
            else:
                if current_state == "training":

                    if current_perceptron.initialized_with_learning_rate:
                        rounded = round(current_perceptron.learning_rate, 4)
                        if rounded > 0:
                            self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                                 text="Lernrate \u03b1: " + str(rounded))
                        else:
                            self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                                 text="Lernrate kleiner als 0.0001")
                    else:
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                             text="")

                    # calculate and plot the weight arrows
                    arrow_dims = self.calc_arrow_dimensions(current_perceptron.weights)
                    self.canvas_perceptron.coords(self.canvas_arrow_id_w_1, 56, 72, 150, 109)
                    self.canvas_perceptron.itemconfigure(self.canvas_arrow_id_w_1, width=arrow_dims[0])

                    self.canvas_perceptron.coords(self.canvas_arrow_id_w_2, 56, 176, 150, 143)
                    self.canvas_perceptron.itemconfigure(self.canvas_arrow_id_w_2, width=arrow_dims[1])
                    if not display_training_step:
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_1,
                                                             text=str(round(current_perceptron.weights[0], 1)),
                                                             fill="blue")
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_2,
                                                             text=str(round(current_perceptron.weights[1], 1)),
                                                             fill="blue")
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_threshold,
                                                             text=str(round(current_perceptron.threshold, 1)),
                                                             fill="brown")
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_1, text="x\u2081")
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2, text="x\u2082")
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_calculated_output, text="Ausgabe")

                    else:
                        if temp_w_x < current_perceptron.weights[0] or temp_w_y < current_perceptron.weights[1]:
                            color = "green"
                        elif temp_w_x > current_perceptron.weights[0] or temp_w_y > current_perceptron.weights[1]:
                            color = "red"
                        else:
                            color = "blue"
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_1,
                                                             text=str(round(temp_w_x, 1)) + " (\u279C " + str(round(
                                                                 current_perceptron.weights[0], 1)) + ")", fill=color)
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_2,
                                                             text=str(round(temp_w_y, 1)) + " (\u279C " + str(round(
                                                                 current_perceptron.weights[1], 1)) + ")", fill=color)
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_threshold,
                                                             text=str(round(temp_threshold, 1)) + " (\u279C " + str(
                                                                 round(
                                                                     current_perceptron.threshold, 1)) + ")",
                                                             fill="brown")

                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_1,
                                                             text=str(training_data_point[0]))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2,
                                                             text=str(training_data_point[1]))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_calculated_output,
                                                             text=str(calculated_output))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_expected_output,
                                                             text="Erwartete Ausgabe: " + str(
                                                                 int(training_data_point[2])))


                elif current_state == "classification":
                    self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                         text="")

                    self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_1,
                                                         text=str(round(current_perceptron.weights[0], 1)), fill="blue")
                    self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_2,
                                                         text=str(round(current_perceptron.weights[1], 1)), fill="blue")
                    self.canvas_perceptron.itemconfigure(self.canvas_text_id_threshold,
                                                         text=str(round(current_perceptron.threshold, 1)),
                                                         fill="brown")
                    if self.main_object.classification_phase == "phase_1" or (
                            classification_first_phase_0 and self.main_object.classification_phase == "phase_0"):
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_1, text="x\u2081")
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2, text="x\u2082")
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_calculated_output,
                                                             text=" ? ")
                    elif self.main_object.classification_phase == "phase_0":
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_1,
                                                             text=str(training_data_point[0]))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2,
                                                             text=str(training_data_point[1]))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_calculated_output,
                                                             text=str(calculated_output))
                        if training_data_point[2] == calculated_output:
                            text_classification_result = "Klassifikation korrekt!"
                            color = "green"
                        else:
                            text_classification_result = "Falsche Klassifikation!"
                            color = "red"
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_expected_output,
                                                             text=text_classification_result, fill=color)
                    elif self.main_object.classification_phase == "phase_2":
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_1,
                                                             text=str(training_data_point[0]))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2,
                                                             text=str(training_data_point[1]))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_calculated_output,
                                                             text=" ? ")
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_expected_output,
                                                             text="Erwartete Ausgabe: " + str(
                                                                 int(training_data_point[2])))
                    elif self.main_object.classification_phase == "phase_3":
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_1,
                                                             text=str(training_data_point[0]))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2,
                                                             text=str(training_data_point[1]))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_calculated_output,
                                                             text=str(calculated_output))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_expected_output,
                                                             text="Erwartete Ausgabe: " + str(
                                                                 int(training_data_point[2])))

                        self.canvas_perceptron.coords(self.canvas_arrow_id_explanation_calculation_weighted_sum, 195,
                                                      35,
                                                      195, 90)
                        self.canvas_perceptron.coords(self.canvas_arrow_id_explanation_calculation_activation_function,
                                                      360,
                                                      65,
                                                      280, 90)

                        weighted_sum = current_perceptron.weights[0] * training_data_point[0] + \
                                       current_perceptron.weights[1] * \
                                       training_data_point[1]

                        explanation_weighted_sum = str(round(current_perceptron.weights[0], 1)) + "\u00B7" + str(
                            training_data_point[0]) + " + " + str(
                            round(current_perceptron.weights[1], 1)) + "\u00B7" + str(
                            training_data_point[1]) + " = " + str(round(weighted_sum, 1))
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_explanation_calculation_weighted_sum,
                                                             text=explanation_weighted_sum)

                        if weighted_sum > current_perceptron.threshold:
                            explanation_activation_function = "Die berechnete Summe ist\n gr\u00F6\u00DFer gleich dem Schwellenwert " + str(
                                round(current_perceptron.threshold, 1))
                        else:
                            explanation_activation_function = "Die berechnete Summe ist\n kleiner als der Schwellenwert " + str(
                                round(
                                    current_perceptron.threshold, 1))
                        self.canvas_perceptron.itemconfigure(
                            self.canvas_text_id_explanation_calculation_activation_function,
                            text=explanation_activation_function)

        elif self.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            # Adapt displaying of perceptron
            self.canvas_perceptron.itemconfig(self.canvas_img_id, image=self.img_perceptron_linear_regression)
            self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_1, text="x")
            self.canvas_perceptron.itemconfigure(self.canvas_text_id_calculated_output, text="Ausgabe")

            self.canvas_perceptron.coords(self.canvas_arrow_id_w_1, 56, 72, 150, 109)
            self.canvas_perceptron.itemconfigure(self.canvas_arrow_id_w_1, width=2)

            # entfernt den Pfeil für w_2, welche nur für die lineare Klassifikation relevant ist
            self.canvas_perceptron.coords(self.canvas_arrow_id_w_2, 0, 0, 0, 0)

            if current_perceptron is None:
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                     text="")

                self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_1,
                                                     text="w\u2081", fill="blue")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2, text=" ")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_2, text=" ")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_threshold,
                                                     text="w\u2080",
                                                     fill="brown")
            else:
                if current_perceptron.initialized_with_learning_rate:
                    rounded = round(current_perceptron.learning_rate, 4)
                    if rounded > 0:
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                             text="Lernrate \u03b1: " + str(rounded))
                    else:
                        self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                             text="Lernrate kleiner als 0.0001")
                else:
                    self.canvas_perceptron.itemconfigure(self.canvas_text_id_learining_rate,
                                                         text="")

                self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_1,
                                                     text=str(round(current_perceptron.weights[0], 5)), fill="blue")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_input_x_2, text=" ")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_w_2, text=" ")
                self.canvas_perceptron.itemconfigure(self.canvas_text_id_threshold,
                                                     text=str(round(current_perceptron.threshold, 5)),
                                                     fill="brown")

    # Regelt die korrekte Anzeige des Plot-Bereichs für den Fall, dass das Perzeptron inintialisiert und Daten geladen sind
    def replot(self, header_data, training_data, current_perceptron, emphasize_point=False,
               data_point=None, point_to_classify=[]):
        """
        Im Modus "Lineare Klassifikation" ist der data_point der aktuell trainierte Datenpunkt
        :param header_data:
        :param training_data:
        :param current_perceptron:
        :param emphasize_point:
        :param data_point:
        :param next_data_point:
        :param point_to_classify:
        """
        if data_point is None:
            data_point = []

        if self.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            if self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE:
                self.plot2D.cla()
            elif self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_WEIGHT_UPDATES:
                self.plot3D.cla()
        elif self.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            if self.selected_display_mode_for_linear_regression.get() == self.mode_LINEAR_REGRESSION_REGRESSION_LINE:
                self.plot2D.cla()
            elif self.selected_display_mode_for_linear_regression.get() == self.mode_LINEAR_REGRESSION_GRADIENT_DESCENT:
                self.plot3D.cla()

        if self.selected_mode.get() == self.mode_LINEAR_CLASSIFICATION:
            if self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_SEPARATION_LINE:
                self.figure_training_data.clear()
                self.figure_training_data.add_axes(self.plot2D)

                if self.show_half_area.get() == 1:
                    my_Plotting.fill(self.plot2D, current_perceptron.weights, current_perceptron.threshold,
                                     self.main_object.plot_dimensions)

                if emphasize_point:
                    my_Plotting.emphasize_point(self.plot2D, data_point)

                # Anzeige der Annotation des nächsten Trainingsdatenpunkts
                # Wenn die Klassifikation eines Punkts aktiviert, kann die Option der Anzeige des nächsten
                # Trainingsdatenpunkts nicht ausgewählt werden. Es wird aber trotzdem der alte Status dieser Anzeige
                # gespeichert. Falls also die Klassifikation aktiviert, wird diese Anzeige ausgeschlossen
                if self.show_next_point_to_train.get() == 1 and not self.classification_point_enabled.get() == 1:
                    if current_perceptron.number_of_training_steps == 0:
                        my_Plotting.annotate_next_training_point(self.plot2D, training_data[0])
                    else:
                        # Aktualisiert den Index des als nächstes zu trainierenden Datums
                        # In call_train wurde bereits trainiert, d.h. die Anzahl der Schritte wurde bereits erhöht
                        # und da der Index bei 0 beginnt, muss die Anzahl nicht mehr um 1 erhöht werden
                        index_of_next_dataset_to_train = (current_perceptron.number_of_training_steps) % (
                            len(training_data))
                        my_Plotting.annotate_next_training_point(self.plot2D,
                                                                 training_data[index_of_next_dataset_to_train])

                # Wenn die Klassifikation aktiviert ist, man sich aber nicht in der Eingabephase des zu klassifizierenden
                # Punkts befindet, wird dieser angezeigt
                if self.classification_point_enabled.get() == 1 and not self.main_object.first_phase_0_after_activation and (
                        self.main_object.classification_phase == "phase_2" or
                        self.main_object.classification_phase == "phase_3" or self.main_object.classification_phase == "phase_0"):
                    my_Plotting.plot_point(self.plot2D, point_to_classify[0], point_to_classify[1])

                # Plotten der akutellen Situation mit Trainingsdaten, Perzeptron und Achsen
                my_Plotting.plotting_data(self.plot2D, header_data,
                                          training_data, self.selected_mode.get(), self.main_object.plot_dimensions)

                my_Plotting.plotting_separation_line(self.plot2D, current_perceptron.weights,
                                                     current_perceptron.threshold, self.main_object.plot_dimensions)
                my_Plotting.plotting_axis_arrows(self.plot2D)
            elif self.selected_display_mode_for_linear_classification.get() == self.mode_LINEAR_CLASSIFICATION_WEIGHT_UPDATES:
                self.figure_training_data.clear()
                self.figure_training_data.add_axes(self.plot3D)
                self.plot3D.dist = 8
                my_Plotting.plotting_gradient_descent_param_updates_lin_classification_with_trace_3D(
                    plot=self.plot3D, plot_accuracy=21, datapoint=data_point, current_perceptron=current_perceptron)

        elif self.selected_mode.get() == self.mode_LINEAR_REGRESSION:
            if self.com_box_lin_regression_display_mode.get() == self.mode_LINEAR_REGRESSION_REGRESSION_LINE:
                self.figure_training_data.clear()
                self.figure_training_data.add_axes(self.plot2D)
                my_Plotting.plotting_data(self.plot2D, header_data,
                                          training_data, self.selected_mode.get(), self.main_object.plot_dimensions)
                my_Plotting.plotting_regression_line(self.plot2D, current_perceptron.weights,
                                                     current_perceptron.threshold, self.main_object.plot_dimensions)
                my_Plotting.plotting_axis_arrows(self.plot2D)
            elif self.com_box_lin_regression_display_mode.get() == self.mode_LINEAR_REGRESSION_GRADIENT_DESCENT:
                self.figure_training_data.clear()
                self.figure_training_data.add_axes(self.plot3D)
                self.plot3D.dist = 8
                # Hier ist der threshold eigentlich ein Bias
                my_Plotting.plotting_loss_surface(self.plot3D, current_perceptron, training_data)
        self.canvas.draw()

    # NICHT VERWENDET
    def calc_arrow_dimensions_old(self, weights):
        """
        Berechnet die Dicke der Gewichts-Pfeile in der Darstellung des Perzeptrons, welche den Einfluss
        der einzelnen Gewichte auf die Ausgabe symbolisieren
        :param weights: Gewichte des Perzeptrons
        :return: Liste der Liniendicken für die Pfeile in der Darstellung des Perzeptron
        """
        max_dimension = 10
        if weights[0] == 0 and weights[1] == 0:
            dim_arrow_x = 1
            dim_arrow_y = 1
        elif weights[0] == 0:
            dim_arrow_x = 1
            dim_arrow_y = max_dimension
        elif weights[1] == 0:
            dim_arrow_x = max_dimension
            dim_arrow_y = 1
        else:
            dim_arrow_x = min(abs(((weights[0] / weights[1]) * max_dimension)), max_dimension)
            dim_arrow_y = min(abs(((weights[1] / weights[0]) * max_dimension)), max_dimension)

        return [dim_arrow_x, dim_arrow_y]

    def calc_arrow_dimensions(self, weights):
        """
        Berechnet die Dicke der Gewichts-Pfeile in der Darstellung des Perzeptrons, welche den Einfluss
        der einzelnen Gewichte auf die Ausgabe symbolisieren.
        Hier wird eine maximale Pfeildicke anteilig auf die einzelnen Gewichte verteilt
        :param weights: Gewichte des Perzeptrons
        :return: Liste der Liniendicken für die Pfeile in der Darstellung des Perzeptron
        """
        max_dimension = 13

        if weights[0] == 0 and weights[1] == 0:
            dim_arrow_x = 1
            dim_arrow_y = 1
        elif weights[0] == 0:
            dim_arrow_x = 1
            dim_arrow_y = max_dimension
        elif weights[1] == 0:
            dim_arrow_x = max_dimension
            dim_arrow_y = 1
        else:
            dim_arrow_x = (abs(weights[0]) / (abs(weights[0]) + abs(weights[1]))) * max_dimension
            dim_arrow_y = (abs(weights[1]) / (abs(weights[0]) + abs(weights[1]))) * max_dimension

        return [dim_arrow_x, dim_arrow_y]

import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QComboBox, QCheckBox,
                             QSpinBox, QLabel, QMainWindow, QAction, QGroupBox, QGridLayout, QSizePolicy, QHBoxLayout)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Make sure to adjust the import statement according to your project structure
from .main import xpute


class XputeThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self, df, impute_zeros,pre_imputation, xgb_models, mf_for_xgb, use_transformed_df,
                 optuna_for_xgb, optuna_n_trials, n_iterations, save_imputed_df, save_plots, test_mode):
        QThread.__init__(self)
        self.df = df
        self.impute_zeros = impute_zeros
        self.pre_imputation = pre_imputation
        self.xgb_models = xgb_models
        self.mf_for_xgb = mf_for_xgb
        self.use_transformed_df = use_transformed_df
        self.optuna_for_xgb = optuna_for_xgb
        self.optuna_n_trials = optuna_n_trials
        self.n_iterations = n_iterations
        self.save_imputed_df = save_imputed_df
        self.save_plots = save_plots
        self.test_mode = test_mode

    # def __del__(self):
    #    self.wait()

    def run(self):
        # simulate a long task
        result = xpute(df=self.df,
                       impute_zeros=self.impute_zeros,
                       pre_imputation=self.pre_imputation,
                       xgb_models=self.xgb_models,
                       mf_for_xgb=self.mf_for_xgb,
                       use_transformed_df=self.use_transformed_df,
                       optuna_for_xgb=self.optuna_for_xgb,
                       optuna_n_trials=self.optuna_n_trials,
                       n_iterations=self.n_iterations,
                       save_imputed_df=self.save_imputed_df,
                       save_plots=self.save_plots,
                       test_mode=self.test_mode)
        self.signal.emit(result)  # we use a signal to return the result


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()

        # Set window title
        self.setWindowTitle("Xputer")

        # Create menu bar
        menu = self.menuBar()

        # Create items for the menu bar
        file_menu = menu.addMenu("Menu")

        # Create actions for the file menu
        open_action = QAction("Load", self)
        open_action.triggered.connect(self.load_csv)  # connect to the load_csv method
        file_menu.addAction(open_action)

        process_action = QAction("Xpute", self)
        process_action.triggered.connect(self.process)  # connect to the process method
        file_menu.addAction(process_action)

        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_csv)  # connect to the export_csv method
        file_menu.addAction(export_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)  # connect to the close method
        file_menu.addAction(exit_action)

        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set fixed size
        #self.setFixedSize(600, 500)  # width, height

        # Grid layout
        self.layout = QGridLayout(self.central_widget)  # set layout on central widget


        # Set a title
        self.title = QLabel("Xputer")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("color: lightblue; font-size: 45px; font-weight: bold;")
        self.layout.addWidget(self.title, 0, 0, 1, 2)  # added to row 0, column 0

        self.kazilab = QLabel("KaziLab.se @ Lund University")
        self.kazilab.setAlignment(Qt.AlignRight)
        self.kazilab.setStyleSheet("color: #3399CC; font-size: 10px; font-weight: bold;")
        self.layout.addWidget(self.kazilab, 0, 2, 1, 2)  # added to row 0, column 0

        self.subtitle = QLabel("An XGBoost powered robust imputer")
        self.subtitle.setAlignment(Qt.AlignRight)
        self.subtitle.setStyleSheet("color: #3399CC; font-size: 12px; font-weight: bold;")
        self.layout.addWidget(self.subtitle, 1, 2, 1, 2)  # added to row 0, column 0

        group_box0 = QGroupBox("Data to be Xputed")  # the argument is the title of the group box
        font0 = QFont()
        font0.setPointSize(10)
        font0.setBold(True)
        group_box0.setFont(font0)
        group_box0.setStyleSheet("QGroupBox::title {color: #4a86e8;}")
        # Create a layout for the group box
        group_layout0 = QVBoxLayout()
        # Create an inner QGridLayout
        inner_layout0 = QGridLayout()


        # Label for load csv file button
        self.load_csv_label = QLabel("Load data file:")
        self.load_csv_label.setAlignment(Qt.AlignRight)
        self.load_csv_label.setToolTip('Click "Load CSV file" button to load a data file. '
                                       'Supported separators: ",", ";" and "\t"')
        self.load_csv_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        #self.layout.addWidget(self.load_csv_label, 1, 0, 1, 1)  # added to row 0, column 0

        # load csv file button
        self.load_csv_button = QPushButton("Load CSV file")
        self.load_csv_button.setStyleSheet("QPushButton { font-size: 12px;}")
        self.load_csv_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.load_csv_button.setMinimumSize(100, 20)  # Set minimum size of QSpinBox (width, height)
        self.load_csv_button.setMaximumSize(100, 20)  # Set maximum size of QSpinBox (width, height)
        self.load_csv_button.setToolTip('Click "Load CSV file" button to load a data file. '
                                       'Supported separators: ",", ";" and "\t"')
        self.load_csv_button.clicked.connect(self.load_csv)
        #self.layout.addWidget(self.load_csv_button, 1, 1, 1, 1)  # added to row 0, column 0, spans 1 row and 2 columns

        # label to display selected file path
        self.file_path_label = QLabel("")
        #self.layout.addWidget(self.file_path_label, 1, 2, 1, 2)  # added to row 1, column 0, spans 1 row and 2 columns

        # Shall we impute zeros
        self.impute_zeros_label = QLabel("Impute zeros:")
        self.impute_zeros_label.setAlignment(Qt.AlignRight)
        self.impute_zeros_label.setToolTip('Check this box if you want to impute zeros')
        self.impute_zeros_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        # self.layout.addWidget(self.mf_for_xgb_label, 3, 0, 1, 1)

        self.impute_zeros = QCheckBox()
        self.impute_zeros.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px;} QCheckBox { font-size: 20px;}")
        # self.layout.addWidget(self.mf_for_xgb, 3, 1, 1, 1)
        self.impute_zeros.setToolTip('Check this box if you want to impute zeros')

        inner_layout0.addWidget(self.load_csv_label, 0, 0, 1, 1)
        inner_layout0.addWidget(self.load_csv_button, 0, 1, 1, 1)
        inner_layout0.addWidget(self.file_path_label, 0, 2, 1, 2)
        inner_layout0.addWidget(self.impute_zeros_label, 1, 0, 1, 1)
        inner_layout0.addWidget(self.impute_zeros, 1, 1, 1, 1)

        # Add inner_layout to the group_layout
        group_layout0.addLayout(inner_layout0)
        # Set the group box's layout
        group_box0.setLayout(group_layout0)
        # Add the group box to the main layout
        self.layout.addWidget(group_box0, 2, 0, 2, 4)

        group_box = QGroupBox("Set parameters (optional)")  # the argument is the title of the group box
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        group_box.setFont(font)
        group_box.setStyleSheet("QGroupBox::title {color: #4a86e8;}")
        # Create a layout for the group box
        group_layout = QVBoxLayout()
        # Create an inner QGridLayout
        inner_layout = QGridLayout()

        # pre_imputation dropdown
        self.pre_imputation_label = QLabel("Initial value to replace NaN:")
        self.pre_imputation_label.setAlignment(Qt.AlignRight)
        # self.pre_imputation_label.setToolTip('Set a method to fill missing data with a preliminary number.')
        self.pre_imputation_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        self.pre_imputation_label.setToolTip('Set a method to fill missing data with a preliminary number.'
                                            '\n we currently support Column mean and KNNImputer')
        # self.layout.addWidget(self.pre_imputation_label, 2, 0, 1, 1)
        self.pre_imputation_dropdown = QComboBox()
        self.pre_imputation_dropdown.setStyleSheet("QComboBox { font-size: 12px;}")
        self.pre_imputation_dropdown.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pre_imputation_dropdown.setMinimumSize(100, 20)  # Set minimum size of QSpinBox (width, height)
        self.pre_imputation_dropdown.setMaximumSize(100, 20)  # Set maximum size of QSpinBox (width, height)
        self.pre_imputation_dropdown.setToolTip('Set a method to fill missing data with a preliminary number.'
                                            '\n we currently support Column mean and KNNImputer')
        self.pre_imputation_dropdown.addItem("MixType")
        self.pre_imputation_dropdown.addItem("ColumnMean")
        self.pre_imputation_dropdown.addItem("KNNImputer")
        # self.layout.addWidget(self.pre_imputation_dropdown, 2, 1, 1, 1)  # added to row 1, column 1

        # xgb_models input
        self.xgb_models_label = QLabel("XGBoost Models:")
        self.xgb_models_label.setAlignment(Qt.AlignRight)
        self.xgb_models_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        self.xgb_models_label.setToolTip('Set a number for XGBoost Models, '
                                 '\n Please set between 3 and 9')

        self.xgb_models = QSpinBox()
        self.xgb_models.setStyleSheet("QSpinBox { font-size: 12px; font-weight: bold;}")
        self.xgb_models.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.xgb_models.setMinimumSize(50, 20)  # Set minimum size of QSpinBox (width, height)
        self.xgb_models.setMaximumSize(50, 20)  # Set maximum size of QSpinBox (width, height)
        self.xgb_models.setMinimum(3)
        self.xgb_models.setMaximum(9)
        # self.xgb_models.setReadOnly(True) # disable manual input
        # self.layout.addWidget(QLabel("xgb_models:"), 3, 0)  # added to row 2, column 0
        # self.layout.addWidget(self.xgb_models, 2, 3, 1, 1)
        self.xgb_models.setToolTip('Set a number for XGBoost Models, '
                                 '\n Please set between 3 and 9')

        # mf_for_xgb dropdown
        self.mf_for_xgb_label = QLabel("Update initial values using NMF:")
        self.mf_for_xgb_label.setAlignment(Qt.AlignRight)
        self.mf_for_xgb_label.setToolTip('Check this box if you want to update initial values, '
                                   '\n using cNMF or SVD (if negative values present in data)')
        self.mf_for_xgb_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        # self.layout.addWidget(self.mf_for_xgb_label, 3, 0, 1, 1)

        self.mf_for_xgb = QCheckBox()
        self.mf_for_xgb.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px;} QCheckBox { font-size: 20px;}")
        # self.layout.addWidget(self.mf_for_xgb, 3, 1, 1, 1)
        self.mf_for_xgb.setToolTip('Check this box if you want to update initial values, '
                                 '\n using cNMF or SVD (if negative values present in data)')

        # use_transformed_df dropdown
        self.use_transformed_df_label = QLabel("Transform full data:")
        self.use_transformed_df_label.setAlignment(Qt.AlignRight)
        self.use_transformed_df_label.setToolTip('Check this box if you want to update full dataframe, '
                                           '\n using cNMF or SVD (if negative values present in data)')
        self.use_transformed_df_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        # self.layout.addWidget(self.use_transformed_df_label, 3, 2, 1, 1)

        self.use_transformed_df = QCheckBox()
        self.use_transformed_df.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px;} QCheckBox { font-size: 20px;}")
        # self.layout.addWidget(self.use_transformed_df, 3, 3, 1, 1)
        self.use_transformed_df.setToolTip('Check this box if you want to update full dataframe, '
                                   '\n using cNMF or SVD (if negative values present in data)')

        # optuna_for_xgb dropdown
        self.optuna_for_xgb_label = QLabel("Hyperparameter search:")
        self.optuna_for_xgb_label.setAlignment(Qt.AlignRight)
        self.optuna_for_xgb_label.setToolTip('Perform a hyperparameter search for XGBoost if checked, can be very slow'
                                             '\n limited to maximum 100 features with missing values, needs minimum '
                                             '\n 50 samples and more than 4-times samples than feature number')
        self.optuna_for_xgb_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        # self.layout.addWidget(self.optuna_for_xgb_label, 4, 0, 1, 1)

        self.optuna_for_xgb = QCheckBox()
        self.optuna_for_xgb.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px;} QCheckBox { font-size: 20px;}")
        # self.layout.addWidget(self.optuna_for_xgb, 4, 1, 1, 1)
        self.optuna_for_xgb.setToolTip('Perform a hyperparameter search for XGBoost if checked, can be very slow'
                                       '\n limited to maximum 100 features with missing values, needs minimum '
                                       '\n 50 samples and more than 4-times samples than feature number')

        # optuna_n_trials input
        self.optuna_n_trials_label = QLabel("Number of trials for Optuna:")
        self.optuna_n_trials_label.setAlignment(Qt.AlignRight)
        self.optuna_n_trials_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        self.optuna_n_trials_label.setToolTip('If hyperparameter search for XGBoost if checked, set a number for trails'
                                        '\n minimum 5 and maximum 50, larger number will make process slower')
        # self.layout.addWidget(self.optuna_n_trials_label, 4, 2, 1, 1)

        self.optuna_n_trials = QSpinBox()
        self.optuna_n_trials.setStyleSheet("QSpinBox { font-size: 12px; font-weight: bold;}")
        self.optuna_n_trials.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.optuna_n_trials.setMinimumSize(50, 20)  # Set minimum size of QSpinBox (width, height)
        self.optuna_n_trials.setMaximumSize(50, 20)  # Set maximum size of QSpinBox (width, height)
        self.optuna_n_trials.setMinimum(5)
        self.optuna_n_trials.setMaximum(50)
        # self.layout.addWidget(QLabel("optuna_n_trials:"), 7, 0)
        # self.layout.addWidget(self.optuna_n_trials, 4, 3, 1, 1)
        self.optuna_n_trials.setToolTip('If hyperparameter search for XGBoost if checked, set a number for trails'
                                           '\n minimum 5 and maximum 50, larger number will make process slower')

        # Add some widgets to the group box's layout
        inner_layout.addWidget(self.pre_imputation_label, 0, 0, 1, 1)
        inner_layout.addWidget(self.pre_imputation_dropdown, 0, 1, 1, 1)
        inner_layout.addWidget(self.xgb_models_label, 0, 2, 1, 1)
        inner_layout.addWidget(self.xgb_models, 0, 3, 1, 1)

        inner_layout.addWidget(self.mf_for_xgb_label, 1, 0, 1, 1)
        inner_layout.addWidget(self.mf_for_xgb, 1, 1, 1, 1)
        inner_layout.addWidget(self.use_transformed_df_label, 1, 2, 1, 1)
        inner_layout.addWidget(self.use_transformed_df, 1, 3, 1, 1)

        inner_layout.addWidget(self.optuna_for_xgb_label, 2, 0, 1, 1)
        inner_layout.addWidget(self.optuna_for_xgb, 2, 1, 1, 1)
        inner_layout.addWidget(self.optuna_n_trials_label, 2, 2, 1, 1)
        inner_layout.addWidget(self.optuna_n_trials, 2, 3, 1, 1)

        # Add inner_layout to the group_layout
        group_layout.addLayout(inner_layout)
        # Set the group box's layout
        group_box.setLayout(group_layout)
        # Add the group box to the main layout
        self.layout.addWidget(group_box, 4, 0, 3, 4)

        group_box1 = QGroupBox("Additional parameters (optional)")  # the argument is the title of the group box
        font1 = QFont()
        font1.setPointSize(10)
        font1.setBold(True)
        group_box1.setFont(font1)
        group_box1.setStyleSheet("QGroupBox::title {color: #4a86e8;}")
        # Create a layout for the group box
        group_layout1 = QVBoxLayout()
        # Create an inner QGridLayout
        inner_layout1 = QGridLayout()

        # How many cycles or iterations to be used
        self.test_mode_label = QLabel("Export intermediate files:")
        self.test_mode_label.setAlignment(Qt.AlignRight)
        self.test_mode_label.setToolTip('Check this box if you want to export intermediate files')
        self.test_mode_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")

        self.test_mode = QCheckBox()
        self.test_mode.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px;} QCheckBox { font-size: 20px;}")
        self.test_mode.setToolTip('Check this box if you want to export intermediate files')

        # optuna_n_trials input
        self.n_iterations_label = QLabel("Number of iterations:")
        self.n_iterations_label.setAlignment(Qt.AlignRight)
        self.n_iterations_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        self.n_iterations_label.setToolTip('If iteration if checked, set a number for iterations'
                                           '\n minimum 1 and maximum 9, larger number will make process slower')

        self.n_iterations = QSpinBox()
        self.n_iterations.setStyleSheet("QSpinBox { font-size: 12px; font-weight: bold;}")
        self.n_iterations.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.n_iterations.setMinimumSize(50, 20)  # Set minimum size of QSpinBox (width, height)
        self.n_iterations.setMaximumSize(50, 20)  # Set maximum size of QSpinBox (width, height)
        self.n_iterations.setMinimum(1)
        self.n_iterations.setMaximum(9)
        self.n_iterations.setToolTip('If iteration if checked, set a number for iterations'
                                     '\n minimum 1 and maximum 9, larger number will make process slower')

        # Shall we save imputed data
        self.save_imputed_df_label = QLabel(" Save result in Document folder:")
        self.save_imputed_df_label.setAlignment(Qt.AlignRight)
        self.save_imputed_df_label.setToolTip('Check this box if you want to save imputed data in your Document folder')
        self.save_imputed_df_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        # self.layout.addWidget(self.mf_for_xgb_label, 3, 0, 1, 1)

        self.save_imputed_df = QCheckBox()
        self.save_imputed_df.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px;} QCheckBox { font-size: 20px;}")
        # self.layout.addWidget(self.mf_for_xgb, 3, 1, 1, 1)
        self.save_imputed_df.setToolTip('Check this box if you want to save imputed data in your Document folder')

        # Shall we create and save plots for each column
        self.save_plots_label = QLabel("Plot imputed values:")
        self.save_plots_label.setAlignment(Qt.AlignRight)
        self.save_plots_label.setToolTip('Check this box if you want to plot imputed values, only continuous data')
        self.save_plots_label.setStyleSheet("QLabel {color: #3399CC; font-size: 12px; font-weight: bold;}")
        # self.layout.addWidget(self.mf_for_xgb_label, 3, 0, 1, 1)

        self.save_plots = QCheckBox()
        self.save_plots.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px;} QCheckBox { font-size: 20px;}")
        # self.layout.addWidget(self.mf_for_xgb, 3, 1, 1, 1)
        self.save_plots.setToolTip('Check this box if you want to plot imputed values, only continuous data')

        inner_layout1.addWidget(self.test_mode_label, 0, 0, 1, 1)
        inner_layout1.addWidget(self.test_mode, 0, 1, 1, 1)
        inner_layout1.addWidget(self.n_iterations_label, 0, 2, 1, 1)
        inner_layout1.addWidget(self.n_iterations, 0, 3, 1, 1)
        inner_layout1.addWidget(self.save_imputed_df_label, 1, 0, 1, 1)
        inner_layout1.addWidget(self.save_imputed_df, 1, 1, 1, 1)
        inner_layout1.addWidget(self.save_plots_label, 1, 2, 1, 1)
        inner_layout1.addWidget(self.save_plots, 1, 3, 1, 1)

        # Add inner_layout to the group_layout
        group_layout1.addLayout(inner_layout1)
        # Set the group box's layout
        group_box1.setLayout(group_layout1)
        # Add the group box to the main layout
        self.layout.addWidget(group_box1, 7, 0, 2, 4)

        # Add the button to the layout with center alignment

        # button to handle the function
        self.process_button = QPushButton("Xpute")
        self.process_button.clicked.connect(self.process)
        self.process_button.setToolTip('Press to run imputation - an input file must be provided to start the process')
        self.process_button.setFixedSize(200, 28)  # width, height
        self.process_button.setStyleSheet("QPushButton {font-size: 15px; font-weight: bold; color: #ffffff; background-color: #3399CC;}")
        container = QWidget()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.process_button, 0, Qt.AlignCenter)
        # Set QHBoxLayout to the container
        container.setLayout(button_layout)
        self.layout.addWidget(container, 9,0,1,2)


        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #3399CC; font-size: 15px; font-weight: bold;")
        self.layout.addWidget(self.status_label, 9, 2, 1, 2)  # You may want to adjust the position

        self.export_csv_button = QPushButton("Export result as CSV")
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_csv_button.setToolTip('Press to export imputed data')
        self.export_csv_button.setFixedSize(200, 28)  # width, height
        self.export_csv_button.setStyleSheet("QPushButton {font-size: 15px; font-weight: bold; color: #ffffff; background-color: lightblue;}")
        container2 = QWidget()
        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(self.export_csv_button, 0, Qt.AlignCenter)
        container2.setLayout(button_layout2)
        self.layout.addWidget(container2, 10,0,1,2)

        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.setFixedSize(200, 28)  # width, height
        self.close_button.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #ffffff; background-color: gray")
        container1 = QWidget()
        button_layout1 = QHBoxLayout()
        button_layout1.addWidget(self.close_button, 0, Qt.AlignCenter)
        container1.setLayout(button_layout1)
        self.layout.addWidget(container1, 10, 2, 1, 2)  # You may want to adjust the position

    def load_csv(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        # update label text with selected file path
        self.file_path_label.setText(f"Loaded file: {self.file_name}")
        if self.file_name:
            print(f"Loaded file {self.file_name}")
        else:
            print("No file was selected")

    def process(self):
        if hasattr(self, 'file_name'):
            separators = [',', ';', '\t']
            df = None
            for sep in separators:
                try:
                    temp_df = pd.read_csv(self.file_name, sep=sep, index_col=0, low_memory=False, skipinitialspace=True)
                    if temp_df.shape[1] > 2:  # Assuming a valid CSV would have at least 3 columns
                        df = temp_df
                        break
                except pd.errors.ParserError:
                    continue
            if df is None:
                raise ValueError("No valid separator detected for the given file.")
            # df = pd.read_csv(self.file_name, index_col=0)
            impute_zeros = self.impute_zeros.isChecked()
            pre_imputation = self.pre_imputation_dropdown.currentText()
            xgb_models = self.xgb_models.value()
            mf_for_xgb = self.mf_for_xgb.isChecked()
            use_transformed_df = self.use_transformed_df.isChecked()
            optuna_for_xgb = self.optuna_for_xgb.isChecked()
            optuna_n_trials = self.optuna_n_trials.value()
            n_iterations = self.n_iterations.value()
            save_imputed_df = self.save_imputed_df.isChecked()
            save_plots = self.save_plots.isChecked()
            test_mode = self.test_mode.isChecked()


            self.status_label.setText("Running...")
            self.process_button.setEnabled(False)  # disable the process button while the task is running

            # start the task in a new thread
            self.thread = XputeThread(df, impute_zeros, pre_imputation, xgb_models, mf_for_xgb, use_transformed_df,
                                      optuna_for_xgb, optuna_n_trials, n_iterations, save_imputed_df,
                                      save_plots, test_mode)

            # connect signals
            self.thread.signal.connect(self.process_finished)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()
            # self.thread.wait()

    def process_finished(self, result):
        self.status_label.setText("Completed - Ready to export result")
        self.process_button.setEnabled(True)  # re-enable the process button
        self.result = result  # save the result

    def export_csv(self):
        if hasattr(self, 'result'):
            # This will open a file dialog to select where to save the file
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if path:
                # If a path is provided (i.e., the dialog isn't cancelled),
                # export the dataframe to a CSV file
                self.result.to_csv(path)
            else:
                print("No file location provided")
        else:
            print("No dataframe to export")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())


def xgui():

    # Create a QApplication instance (needed for any GUI in PyQt)
    app = QApplication([])

    window = Window()
    window.show()

    # Start the application's event loop
    app.exec_()

    # Start the application's event loop
    app.exec_()

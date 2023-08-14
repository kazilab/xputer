"""
******************
***   Xputer  ***
******************

An XGBoost powered robust imputer.
"""
__author__ = 'Julhash Kazi'
__email__ = 'xputer@kazilab.se'
__description__ = 'xputer'
__version__ = '0.2.0'
__url__ = 'https://www.kazilab.se'

from .main import Xpute, xpute, preprocessing_df, cnmf, run_svd
from .gui import xgui

Xputer is developed for research purposes only.
The application has been tested using small datasets.
Although it performed well, it does not guarantee true imputations.
The application has no guarantee and must be used with the user's responsibility.

We recommend using the Anaconda environment (install a recent copy of Anaconda from https://www.anaconda.com). Run Anaconda Powershell Prompt and use "pip install xputer" for installation.

To run the application GUI use "python -c "from xputer import xgui".

Jupyter Notebook script
from xputer import xgui
xgui()


To incorporate a pipeline: 
from xputer import Xpute

data = A pandas data frame (sample name as index and features as headers).

imputed_data = Xpute().fit(data)

or 

from xputer import xpute
imputed_data = xpute(data)

-----------------------------
On Windows:
python setup.py install

On Linux:
sudo python setup.py install 

On Mac OS X:
sudo python ./setup.py install 


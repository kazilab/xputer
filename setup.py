"""
Setup alphaML
"""
import sys
try:
    from setuptools import setup, find_packages
except ImportError:
    raise ImportError("Please install `setuptools`")

if not (sys.version_info[0] == 3 and sys.version_info[1] >= 9):
    raise RuntimeError(
                'alphaML requires Python 3.9 or higher.\n'
                'You are using Python {0}.{1}'.format(
                    sys.version_info[0], sys.version_info[1])
                )

# main setup command
setup(
    name='xputer',
    version='0.2.0', # Major.Minor.Patch
    author='Julhash Kazi',
    author_email='xputer@kazilab.se',
    url='https://www.kazilab.se',
    description='An XGBoost powered robust imputer',
    license='Apache-2.0',
    install_requires=[
        'IPython',
        'joblib',
        'matplotlib',
        'nimfa',
        'numpy',
        'optuna',
        'pandas',
        'scikit-learn',
        'torch',
        'tqdm',
        'xgboost'
    ],
    extras_require={
        'gui': ['PyQt5']
    },
    platforms='any',
    packages=find_packages()
)

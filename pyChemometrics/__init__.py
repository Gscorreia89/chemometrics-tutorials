from .ChemometricsPCA import ChemometricsPCA
from .ChemometricsPLS import ChemometricsPLS
from .ChemometricsScaler import ChemometricsScaler
from .ChemometricsPLSDA import ChemometricsPLSDA
from .ChemometricsOrthogonalPLS import ChemometricsOrthogonalPLS

__version__ = '0.1'

__all__ = ['ChemometricsScaler', 'ChemometricsPCA', 'ChemometricsPLS',
           'ChemometricsPLSDA', 'ChemometricsOrthogonalPLS']

"""

The pyChemometrics module provides objects which wrap pre-existing scikit-learn PCA and PLS algorithms and adds 
model some of the common routines and model assessment metrics seen in the Metabolomics and Chemometrics literature.

"""

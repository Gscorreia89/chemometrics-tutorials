# Chemometric data analysis tutorials

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Gscorreia89/chemometrics-tutorials/master)

This repository contains a series of tutorials on multivariate analysis of metabolic profiling datasets: 
 - Import, scaling & normalisation.ipynb: Introduction to normalisation, scaling and data transformations.  
 - Multivariate Analysis - PCA.ipynb: Multivariate chemometric analysis using Principal Component Analysis.
 - Multivariate Analysis - Supervised Analysis with PLS-DA.ipynb: Discrimnination of 2 classes with PLS-DA.
 - Univariate Analysis.ipynb: Univariate analysis with linear models 
 
To run these tutorials download the contents of this repository and run the Jupyter Notebooks. Alternatively, these can be run on the browser via Binder, by clicking on the sticker above (subject to load on the Binder servers). 

All the data required to run these notebooks is provided on the 'data' folder. The dataset used in this tutorial comes from the following publication:
- Blaise, Benjamin J. et al. “Metabotyping of Caenorhabditis elegans reveals latent phenotypes.” Proceedings of the National Academy of Sciences of the United States of America 104 50 (2007): 19808-12 .

It is a set of 139 proton high-resolution magic angle spinning NMR spectroscopy (1H HR-MAS NMR) spectra from 
C. elegans nematodes. There are two main biological sources of variation in the dataset:
- Genotype (0: wild-type, 1: *sod-2* mutants)
- Age/life stage (0: younger L2 worms, 1: L4 worms)


## Installation instructions:
To run these tutorials, first download and install the latest Anaconda Python 3.x distribution available. 
The following packages also need to be installed:
 - plotly: Available via conda (open an Anaconda prompt and type "conda install plotly")
After installing plotly, either clone this repository or download as .zip file. Access the notebooks contents via the Jupyter notebook environment. 

All other dependencies required to run the tutorials (pyChemometrics toolbox) are bundled with the repository, and no specific installation is required. 

For more information on how to use Jupyter notebooks [Using the Jupyter notebook](https://docs.anaconda.com/ae-notebooks/user-guide/basic-tasks/apps/jupyter/)


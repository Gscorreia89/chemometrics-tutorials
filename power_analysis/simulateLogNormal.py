import numpy as np
from sklearn.covariance import LedoitWolf, OAS
import sys


def simulateLogNormal(data, covtype='Estimate', nsamples=2000, **kwargs):
    """

    :param data:
    :param covtype: Type of covariance matrix estimator. Allowed types are:
        - Estimate (default):
        - Diagonal:
        - Shrinkage OAS:
    :param int nsamples: Number of simulated samples to draw
    :return: simulated data and empirical covariance est
    """

    try:
        # Offset data to make sure there are no 0 values for log transform
        offset = np.min(data) + 1
        offdata = data + offset

        # log on the offsetted data
        logdata = np.log(offdata)
        # Get the means
        meanslog = np.mean(logdata, axis=0)

        # Specify covariance
        # Regular covariance estimator
        if covtype == "Estimate":
            covlog = np.cov(logdata, rowvar=0)
        # Shrinkage covariance estimator, using LedoitWolf
        elif covtype == "ShrinkageLedoitWolf":
            scov = LedoitWolf()
            scov.fit(logdata)
            covlog = scov.covariance_
        elif covtype == "ShrinkageOAS":
            scov = OAS()
            scov.fit(logdata)
            covlog = scov.covariance_

        # Diagonal covariance matrix (no between variable correlation)
        elif covtype == "Diagonal":
            covlogdata = np.var(logdata, axis=0)       #get variance of log data by each column
            covlog = np.diag(covlogdata)               #generate a matrix with diagonal of variance of log Data
        else:
            raise ValueError('Unknown Covariance type')

        simData = np.random.multivariate_normal(meanslog, covlog, nsamples)
        simData = np.exp(simData)
        simData -= offset

        ##Set to 0 negative values
        simData[np.where(simData < 0)] = 0
        # work out the correlation of matrix by columns, each column is a variable
        corrMatrix = np.corrcoef(simData, rowvar=0)

        return simData, corrMatrix

    except Exception as exp:
        raise exp


if __name__ == "__main__":
    simData, corrMatrix = simulateLogNormal(sys.argv[1:])
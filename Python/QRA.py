from quantreg import quantreg
from multiprocessing import Pool
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

data = pd.read_csv('lear_forecast.csv', index_col=0) # contains 5 forecast columns and real price
data.loc[data.index[:], 'ones'] = 1.0
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]

if not os.path.exists(os.path.join('..', 'lear_QRA')):
    os.mkdir(os.path.join('..', 'lear_QRA'))

# on avg
def quantreg_day(dno):
    """
    Computes and saves the 24-hour probabilistic price forecast (99 percentiles)
    for a single out-of-sample test day using Quantile Regression Averaging (QRA).

    Args:
        dno (int): Zero-based index of the out-of-sample test day (0 to 553),
            corresponding to calendar day `dno + 182` of the 736-day forecast period.

    Process:
        1. Slices the preceding 182 days (4,368 hourly rows) as the rolling QR calibration window.
        2. Fits 99 pooled linear quantile regressions (tau = 0.01..0.99) mapping the 4 LEAR
           point forecasts + intercept to realized prices across all 24 hours.
        3. Applies the fitted weights to day `dno + 182`'s 24-hour point forecasts and sorts
           each hour's 99 quantiles ascending to prevent quantile crossing.
        4. Saves the resulting (99, 24) quantile matrix to `../lear_QRA/<YYYY-MM-DD>`.
    """
    # dno is range(736-182)
    X = data.iloc[dno*24:dno*24+182*24] # take the training window for the QR of 182 days
    d = datetime.strftime(data.index[dno*24+182*24+1], '%Y-%m-%d') 
    Xf = data.iloc[dno*24+182*24:dno*24+183*24][['forecast56', 'forecast84', 'forecast1092', 'forecast1456', 'ones']].to_numpy() # next day to forecast
    Y = X['real'].to_numpy() # QR window prices
    X = X[['forecast56', 'forecast84', 'forecast1092', 'forecast1456', 'ones']].to_numpy() # QR window features
    taus = [i/100 for i in range(1,100)] # quantile definition
    betas = {}
    for t in taus:
        betas[t] = quantreg(X, Y, t)
    prob = np.zeros((len(taus), 24))
    for h in range(24):
        h_res = []
        for t in taus:
            h_res.append(np.dot(Xf[h, :], betas[t])) # dot product to get the quantile prediction
        # print(h_res)
        h_res.sort()
        prob[:, h] = h_res
    np.savetxt(os.path.join('..', 'lear_QRA', d), prob, delimiter=',', fmt='%.3f')

with Pool() as p:
    _ = p.map(quantreg_day, list(range(736-182)))

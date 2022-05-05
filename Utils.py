import os
import sys
import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib as plot
import itertools
from tqdm import tqdm

def convert_to_daily_returns(closing_prices):
    return closing_prices.pct_change()


def convert_to_total_returns(daily_rets):
    return (daily_rets + 1).cumprod() - 1


def calc_mu(daily_rets, center_mass):
    return daily_rets.ewm(com=center_mass).mean().iloc[-1]


def calc_cov(daily_rets, center_mass):
    return daily_rets.ewm(com=center_mass).cov().iloc[-5:]


def calc_sharpe(rets):
    return (rets.mean()[0] / rets.std()[0]) * np.sqrt(252)

def scale_to_vol(rets, target_vol=.1):
    return rets * (target_vol / rets.std())
import pandas as pd
import numpy as np
import math

from scipy.stats import *
from scipy.signal import periodogram, find_peaks
from numpy.fft import fft, fftfreq, ifft

from hurst import compute_Hc


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from statsmodels.tsa.stattools import OLS
from scipy.interpolate import CubicSpline

from statsmodels.stats.anova import anova_lm
import matplotlib.dates as mdates

from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn')





def subsetting(data: pd.Series, k: int) -> list:
        
        def chunks(l, n):
      
            for i in range(0, len(l), n): 
                yield l[i:i + n]

        return list(chunks(data, k))
    
    
def date_parser(timestamp):
    return np.round(timestamp.year * (timestamp.month / 12) * (timestamp.day / 30) * (timestamp.hour / 24),2)

def log_return(series: pd.Series, period: int) -> pd.Series:
    
    new_series = series / series.shift(period)
    new_series.dropna(inplace = True)
    log_ret = np.log(new_series)
    log_ret[log_ret == np.inf] = np.NaN
    log_ret[log_ret == -np.inf] = np.NaN
    log_ret.dropna(inplace = True)
    
    return log_ret

def print_moments(series: pd.Series) -> float:
    
    mu = series.mean()
    sigma = series.std()
    skew = series.skew()
    kurt = series.kurtosis()
    excess_kurt = kurt - 3
    
    print('Sample Data Distribution Moments')
    print('='*80)
    print('Sample Mean: {:.2f}'.format(mu))
    print('-'*80)
    print('Sample Standard Deviation: {:.2f}'.format(sigma))
    print('-'*80)
    print('Sample skewness: {:.2f}'.format(skew))
    print('-'*80)
    print('Sample Kurtosis: {:.2f}'.format(kurt))
    print('-'*80)
    print('Excess of Kurtosis: {:.2f}'.format(excess_kurt))
    print('='*80)
    
    


def create_panel_data(series: pd.Series) -> list:
    
    panel_data = []
    
    for frame in series.groupby([series.index.hour, series.index.minute]):
        panel_data.append(frame[1])
        
    return panel_data

def create_panel_df(series: pd.Series) -> pd.DataFrame:
    
    panel_data = create_panel_data(series)
    indexing = panel_data[0].index.floor('D').strftime('%Y-%m-%d')
    
    panel_list = []
    
    for panel in panel_data:
        
        raw_panel = panel.reset_index(drop=True)
        panel_list.append(raw_panel)
        
    panel_df = pd.concat([panel_list[k] for k in range(len(panel_list))], axis=1)
    panel_df.index = indexing
    panel_df.columns = list(range(len(panel_list)))
    
    return panel_df



def dickey_fuller_test(data:pd.Series, confidence_interval = 0.05):
    
    test = adfuller(data)
    stats = float(test[0])
    pvalue = float(test[1])
    one_pct = float(test[4]['1%'])
    
    h0 = (stats < one_pct) and (pvalue < confidence_interval) 
    
    print('Augmented Dickey-Fuller test')
    print('='*35)
    print('Test statistic: {}'.format(round(float(test[0]),4)))
    print('-'*35)
    print('p-value: {}'.format(float(test[1])))
    print('-'*35)
    print('Critical Values:')
    for key, value in test[4].items():
        print('\t%s: %.3f' % (key, float(value)))
    print('='*35)
    print('Hypothesis Testing Results')
    print('='*35)
    print('Null Hyptothesis (H0): {}'.format(bool(h0)))
    print('='*35)

def kpss_test(series, confidence_interval = 0.05):    
    statistic, p_value, n_lags, critical_values = kpss(series)
    # Format Output
    print('Kwiatkowski-Phillips-Schmidt-Shin test')
    print('='*35)
    print(f'Test statistic: {statistic}')
    print('-'*35)
    print(f'p-value: {p_value}')
    print('-'*35)
    print(f'num lags: {n_lags}')
    print('-'*35)
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print('Hypothesis Testing Results')
    print('='*35)
    h0 = (statistic < critical_values['1%']) & (p_value < confidence_interval)
    print(f'Result: The series is {"not " if h0 else ""}stationary')





def rolling_hurst(series: pd.Series, lags:int, plot = False) -> pd.Series:
    
    if not isinstance(series, pd.DataFrame):
        series = series.to_frame('value')
    
    if lags < 100:
        raise ValueError(f'To compute the Hurst Exponent we need at least 100 lags, found {lags}')
    
    hr = series.rolling(window = lags).apply(lambda x : compute_Hc(x)[0])

    if plot:
        
        hr.plot(title = 'Rolling Hurst Exponent', figsize = (16,9), grid = True)
    
    else:
        
        return hr.dropna()
    
    
def rolling_correlation(data:pd.DataFrame, window:int) -> pd.Series:
    
    return data[data.columns[0]].rolling(window=window).corr(data[data.columns[1]], method = 'pearson')
    
    
    
    
class DetectSeasons():
    
    def __init__(self, data:pd.Series, size:int, period:int):
        
        self.data = data
        self.size = size
        self.period = period
        
        if self.size > self.data.shape[0]:
            raise ValueError('size should not exceed the length of the series')
        
        self.fourier_filter()
    
    def fourier_filter(self):
        
        self.signal = self.data[:self.size] - self.data[:self.size].mean()
        n = self.signal.shape[0]
        
        
        self.fft_output = fft(self.signal)
        power = abs(self.fft_output)
        freq = fftfreq(n)
        
        mask = freq > 0
        self.freq = freq[mask]
        self.power = power[mask]
        
        
        self.peaks = find_peaks(self.power, prominence=10**4)[0]
        self.peak_freq =  self.freq[self.peaks]
        self.peak_power = self.power[self.peaks]
        
        
    def result(self) -> pd.DataFrame:
        
        output = pd.DataFrame()
        output['index'] = self.peaks
        output['freq (1/hour)'] = self.peak_freq
        output['amplitude'] = self.peak_power
        output[f'period {self.period}'] = np.round(1 / self.peak_freq / self.period, 2)
        output['fft'] = self.fft_output[self.peaks]
        output = output.sort_values('amplitude', ascending=False)
        
        return output
        
        
    def plot_result(self):
        
        plt.figure(figsize=(16, 8))

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(self.freq, self.power, label='signal')
        ax1.set_title('All Frequencies')
        ax1.set_ylabel('Amplitude' )
        ax1.set_xlabel('Frequency')
        plt.xticks(rotation=90)

        ax2 = plt.subplot( 1, 2, 2 )
        mask = (self.freq > 0) & (self.freq <= 0.5)
        ax2.plot(self.freq[mask], self.power[mask])
        ax2.set_title('Frequencies in (0, 0.5]')
        ax2.set_ylabel( 'Amplitude' )
        ax2.set_xlabel( 'Frequency ' )

        plt.plot(self.peak_freq, self.peak_power, 'ro')

        plt.tight_layout()
        plt.xticks(rotation=90)
        
        
class SeasonalWavelets():
    
    def __init__(self, data:pd.Series, multiplier:int, frequencies:list):
        
        self.data = data
        self.multiplier = multiplier
        self.frequencies = frequencies
        self.value_range = pd.Series(np.linspace(self.data.min(), self.data.max(), self.data.shape[0]), index = self.data.index)
        
        
    def fit(self) -> pd.Series:
        
        self.calibration()
        
        result = self.result
        
        best_sigma = float(result.head(1).index.values)
        
        return self.wavelet(sigma = best_sigma)
    
    
    def wavelet(self, sigma = 1):
        
        multiplier = self.multiplier
        frequencies = self.frequencies
        value_range = self.value_range
    
        lst_values = []
    
        np.random.seed(123)
        
        noise = np.random.normal(0, sigma, len(value_range))
        
        for frequency in frequencies:
            lst_values.append(multiplier*np.cos(noise)*np.cos((2*np.pi*value_range/frequency)) + multiplier*np.cos(noise)*np.sin(2*np.pi*value_range/frequency))
                            
        return sum(lst_values)
    
    
    def calibration(self):
        
        data = self.data.copy()
        value_range = self.value_range
        
        
        calibration = []
        
        for sigma in np.arange(0.0, 10.0, 0.01).tolist():
            y_hat = self.wavelet(sigma=sigma)
            mse = mean_absolute_error(y_hat, data)
            rmse = mean_squared_error(y_hat, data, squared = False)
            calibration.append((sigma, mse, rmse))
            
        result = pd.DataFrame(data=calibration, columns = ['sigma', 'mse', 'rmse'])
        result.set_index('sigma', drop = True, inplace = True)

        self.result = result.sort_values(by = 'rmse')
            
    
    
    
class WaveletApproximation():
    
    def __init__(self, multiplier:int, frequencies:list):
        
        self.multiplier = multiplier
        self.frequencies = frequencies
        
        
    def fit(self, X, y):
        
        self.param_estimation(X=X, y=y)
        
        result = self.result
        
        self.best_sigma = float(result.head(1).index.values)
        
        self.fitted_values = self.wavelets(X, sigma=self.best_sigma)
        
        
    def predict(self, X):
        
        if self.fit is None:
            print('Fit the model first')
            
        value_range = pd.Series(date_parser(X.index), index = X.index)
        
        return self.wavelets(X, sigma = self.best_sigma)
        
        
        
    
    def wavelets(self, X, sigma):
        
        value_range = pd.Series(date_parser(X.index), index = X.index)
        multiplier = self.multiplier
        frequencies = self.frequencies
        
        noise = np.random.normal(0, sigma, len(value_range))
        
        lst_values = []
        
        for frequency in frequencies:
            lst_values.append(multiplier*np.cos(noise)*np.cos((2*np.pi*value_range/frequency)) + multiplier*np.cos(noise)*np.sin(2*np.pi*value_range/frequency))
                            
        return sum(lst_values)
    
    
    # Need MLE for the fit function, want to estimate multiplier, sigma 
    
    
    def param_estimation(self, X, y):
        
        value_range = pd.Series(date_parser(X.index), index = X.index)
        
        sigma_range = np.arange(0, 20, 0.01)
        
        calibration = []
        
        for sigma in sigma_range.tolist():
            y_hat = self.wavelets(value_range, sigma=sigma)
            mse = mean_absolute_error(y_hat, y)
            rmse = mean_squared_error(y_hat, y, squared = False)
            
            calibration.append((sigma, mse, rmse))
            
        result = pd.DataFrame(data=calibration, columns = ['sigma', 'mse', 'rmse'])
        result.set_index('sigma', drop = True, inplace = True)

        self.result = result.sort_values(by = 'rmse')
        
        
        
        
        
#Bootstrapped Prediction Interval

def bootstrap_prediction_interval(y_train: pd.Series,
                                  y_fit: pd.Series,
                                  y_pred_value: float,
                                  alpha: float = 0.05,
                                  nbootstrap: int = None,
                                  seed: int = None):
    
    n = len(y_train)

    # compute the forecast errors/resid
    fe = y_train - y_fit

    # get percentile bounds
    percentile_lower = (alpha * 100) / 2
    percentile_higher = 100 - percentile_lower

    if nbootstrap is None:
        nbootstrap = np.sqrt(n).astype(int)
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    # bootstrap sample from errors
    error_bootstrap = []
    for _ in range(nbootstrap):
        idx = rng.integers(low=n)
        error_bootstrap.append(fe[idx])

    # get lower and higher percentiles of sampled forecast errors
    fe_lower = np.percentile(a=error_bootstrap, q=percentile_lower)
    fe_higher = np.percentile(a=error_bootstrap, q=percentile_higher)

    # compute P.I.
    pi = [y_pred_value + fe_lower, y_pred_value, y_pred_value + fe_higher]

    return pi



# Mean Reversion half life signals

def halflife(data:pd.Series) -> float:
    
    diff1 = data.diff().dropna()
    const = sm.add_constant(data.shift(1)).dropna()
    model = sm.OLS(diff1, const)
    res = model.fit()
    
    return -np.log(2) / res.params[1]

def rolling_halflife(data: pd.Series, window: int, plot = False) -> pd.Series:
    
    rol = list(data.rolling(window = window))
    
    hl_lst = []
    
    for rl in rol:
        
        if len(rl) >= window:
            
            hl_lst.append(halflife(rl))
            
    
            
    hl = pd.Series(data=hl_lst, index = data.index[window-1:])
        
    if plot:
        
        hl.plot(figsize = (16,8), title = 'Rolling Half-Life of mean reversion')
        
   
    else:
            
        return hl.dropna()
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

plt.style.use('seaborn')



class StrategyEvaluator():
    
    def __init__(self, data: pd.Series, rounding: int):
        
        self.data = data.to_frame('sreturns')
        self.data.dropna(inplace=True)
        self.rounding = rounding
        
        self.trades = len(data)
        
        self.get_performance()
        
        
        
    def print_performance(self):
        
        results = self.results
        
        print('PERFORMANCE METRICS')
        print('='*80)
        print('Terminal Wealth: {}'.format(results['terminal_wealth']))
        print('-'*80)
        print('Sharpe Ratio: {}'.format(results['sharpe_ratio']))
        print('-'*80)
        print('Sortino Ratio: {}'.format(results['sortino_ratio']))
        print('-'*80)
        print('Hit Rate: {}'.format(results['hit_rate']))
        print('-'*80)
        print('# Trades: {} | Wins: {} | Losses: {}'.format(self.trades, results['winners'], results['loosers']))
        print('-'*80)

        print('Average Return: {}'.format(results['avg_return']))
        print('-'*80)
        print('Median Return: {}'.format(results['median_return']))
        print('-'*80)
        print('Standard Deviation of Returns: {}'.format(results['standard_deviation']))
        print('-'*80)

        print('HHI: {} | HHI plus: {} | HHI minus: {}'.format(round(results['hhi_np'], self.rounding), round(results['hhi_pos'], self.rounding), round(results['hhi_neg'], self.rounding)))
        print('-'*80)
        print('Gain-Loss Ratio: {}'.format(results['gainloss_ratio']))
        print('-'*80)
        print('5%: {} | 33%: {} | 50%: {} | 67%: {} | 95%: {}'.format(results['q05'], results['q33'], results['q50'], results['q67'], results['q95']))
        print('-'*80)

        print('='*80)
        
    
    def plot_performance(self) -> plt.plot:
        
        return self.total_returns().plot(figsize=(16,8), title = 'Strategy Equity Curve')
    
    def plot_distribution(self) -> plt.plot:
        
        sns.set(rc={'figure.figsize':(16.8,9.27)})
        sns.histplot(self.data.sreturns, kde=True)
    
        
    def get_performance(self) -> pd.Series:
        
        terminal = self.total_returns()[-1]
        sharpe = self.sharpe_ratio()
        sortino = self.sortino_ratio()
        glos = self.gainloss_ratio()
        
        trades = self.trades
        hr = self.hit_rate()
        
        winners = self.winners()
        loosers = self.loosers()
        
        avg_rets = self.mean_returns()
        mrets = self.median_returns()
        stdev = self.stdev_returns()
        
        hhi_pos = round(self.hhi(self.data.query('sreturns > 0').sreturns), self.rounding)
        hhi_neg = round(self.hhi(self.data.query('sreturns < 0').sreturns), self.rounding)
        hhi_np = round(self.hhi(self.data.sreturns), self.rounding)
        
        quants = self.quantiles()
        
        q05 = quants[0.05]
        q33 = quants[0.33]
        q50 = quants[0.50]
        q67 = quants[0.67]
        q95 = quants[0.95]
        
        all_info = [terminal, sharpe, sortino, glos, trades, hr, winners, loosers, 
                    avg_rets, mrets, stdev, hhi_pos, hhi_neg, hhi_np, q05, q33, q50, q67, q95]
        
        infos = pd.Series(all_info)
        
        infos.index = ['terminal_wealth', 'sharpe_ratio', 'sortino_ratio', 'gainloss_ratio', 'n_trades',
                       'hit_rate', 'winners', 'loosers', 'avg_return', 'median_return', 'standard_deviation',
                       'hhi_pos', 'hhi_neg', 'hhi_np', 'q05', 'q33', 'q50', 'q67', 'q95']
        
        self.results = infos
        
            
    
        
    def total_returns(self) -> float:
        return round(self.data.sreturns.cumsum(), self.rounding)
    
    def sharpe_ratio(self) -> float:
        return round(self.data.query('sreturns != 0').sreturns.mean() / self.data.query('sreturns != 0').sreturns.std(), self.rounding)
    
    def sortino_ratio(self) -> float:
        return round(self.data.query('sreturns > 0').sreturns.mean() / self.data.query('sreturns < 0').sreturns.std(), self.rounding)
    
    def gainloss_ratio(self) -> float:
        return round(self.data.sreturns.max() / abs(self.data.sreturns.min()), self.rounding)
    
    def hit_rate(self) -> float:
        return round(self.data.query('sreturns > 0').shape[0] / self.trades, self.rounding)
    
    def winners(self) -> int:
        return self.data.query('sreturns > 0').shape[0]
    
    def loosers(self) -> int:
        return self.trades - self.winners()
    
    def mean_returns(self) -> float:
        return round(self.data.sreturns.mean(), self.rounding)
    
    def median_returns(self) -> float:
        return round(self.data.sreturns.median(), self.rounding)
    
    def stdev_returns(self) -> float:
        return round(self.data.sreturns.std(), self.rounding)
    
    def hhi(self, betRet) -> float:
    
        if betRet.shape[0]<= 2: return np.NaN

        wght = betRet / betRet.sum()

        hhi = (wght**2).sum()

        hhi=(hhi-betRet.shape[0]**-1)/(1.-betRet.shape[0]**-1)

        return hhi
    
    def quantiles(self) -> float:
        
        qs = [0.05, 0.33, 0.5, 0.67, 0.95]
        
        return round(self.data.sreturns.quantile(qs), self.rounding)
    
        
        
        
        
    

        
        
        
        
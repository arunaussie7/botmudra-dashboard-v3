import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import MetaTrader5 as mt5

class ForexBasketIndexStrategy:
    def __init__(self, pairs=None, ema_period=200, risk_per_trade=2.0):
        # Default pairs without 'm' suffix
        self.base_pairs = pairs or ["GBPUSD", "EURUSD", "NZDUSD", "AUDUSD"]
        self.pairs = []  # Will be populated with correct symbols (with or without 'm' suffix)
        self.ema_period = ema_period
        self.risk_per_trade = risk_per_trade
        self.basket_data = None
        self.ema_line = None
        
        # Initialize MT5 connection if not already connected
        if not mt5.initialize():
            mt5.shutdown()
            if not mt5.initialize():
                raise Exception("Failed to connect to MetaTrader5")
        
        # Get all available symbols
        symbols = mt5.symbols_get()
        available_symbols = [symbol.name for symbol in symbols] if symbols else []
        
        # Find the correct symbol names (with or without 'm' suffix)
        for pair in self.base_pairs:
            if f"{pair}m" in available_symbols:
                self.pairs.append(f"{pair}m")
            elif pair in available_symbols:
                self.pairs.append(pair)
            else:
                raise ValueError(f"Symbol {pair} not found in MT5")
                
        print(f"Initialized strategy with pairs: {self.pairs}")
        
    def calculate_basket_index(self, data_dict):
        """Calculate the basket index from individual pair prices"""
        df_list = []
        for pair in self.pairs:
            if pair in data_dict:
                df = data_dict[pair].copy()
                df['close'] = df['close'] / df['close'].iloc[0]  # Normalize prices
                df_list.append(df['close'])
        
        if not df_list:
            return None
            
        # Calculate basket index as average of normalized prices
        basket_index = pd.concat(df_list, axis=1).mean(axis=1)
        return basket_index
        
    def calculate_ema(self, series):
        """Calculate Exponential Moving Average"""
        return series.ewm(span=self.ema_period, adjust=False).mean()
        
    def get_historical_data(self, timeframe, start_date, end_date):
        """Fetch historical data for all pairs"""
        data_dict = {}
        
        for pair in self.pairs:
            # Convert timeframe string to MT5 timeframe
            mt5_timeframe = {
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }.get(timeframe)
            
            if not mt5_timeframe:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            # Fetch historical data
            rates = mt5.copy_rates_range(pair, mt5_timeframe, start_date, end_date)
            if rates is None or len(rates) == 0:
                print(f"No data available for {pair}")
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            data_dict[pair] = df
            
        return data_dict
        
    def backtest(self, timeframe, start_date, end_date, initial_balance=10000):
        """Run backtest for the strategy"""
        # Get historical data
        data_dict = self.get_historical_data(timeframe, start_date, end_date)
        if not data_dict:
            raise ValueError("No data available for backtesting")
            
        # Calculate basket index
        self.basket_data = self.calculate_basket_index(data_dict)
        if self.basket_data is None:
            raise ValueError("Failed to calculate basket index")
            
        # Calculate EMA
        self.ema_line = self.calculate_ema(self.basket_data)
        
        # Initialize results
        balance = initial_balance
        trades = []
        positions = []
        equity_curve = [initial_balance]
        dates = [self.basket_data.index[0]]
        
        # Run backtest
        for i in range(1, len(self.basket_data)):
            current_time = self.basket_data.index[i]
            
            # Calculate signals
            prev_basket = self.basket_data[i-1]
            curr_basket = self.basket_data[i]
            prev_ema = self.ema_line[i-1]
            curr_ema = self.ema_line[i]
            
            # Check for position exits on opposite signals
            if positions:
                # Close long positions on sell signal
                if prev_basket >= prev_ema and curr_basket < curr_ema and any(p['type'] == 'buy' for p in positions):
                    for pos in positions[:]:  # Copy list for safe iteration
                        if pos['type'] == 'buy':
                            exit_price = data_dict[pos['pair']]['close'][i]
                            profit_pips = (exit_price - pos['entry']) * 10000
                            profit_amount = (profit_pips / 100) * pos['risk_amount']  # Scale profit by risk amount
                            balance += profit_amount
                            
                            trades.append({
                                'date': current_time,
                                'type': pos['type'],
                                'pairs': self.pairs,
                                'entry': pos['entry'],
                                'exit': exit_price,
                                'profit': profit_amount,
                                'balance': balance
                            })
                            
                            positions.remove(pos)
                
                # Close short positions on buy signal
                elif prev_basket <= prev_ema and curr_basket > curr_ema and any(p['type'] == 'sell' for p in positions):
                    for pos in positions[:]:  # Copy list for safe iteration
                        if pos['type'] == 'sell':
                            exit_price = data_dict[pos['pair']]['close'][i]
                            profit_pips = (pos['entry'] - exit_price) * 10000
                            profit_amount = (profit_pips / 100) * pos['risk_amount']  # Scale profit by risk amount
                            balance += profit_amount
                            
                            trades.append({
                                'date': current_time,
                                'type': pos['type'],
                                'pairs': self.pairs,
                                'entry': pos['entry'],
                                'exit': exit_price,
                                'profit': profit_amount,
                                'balance': balance
                            })
                            
                            positions.remove(pos)
            
            # Check for new position entry
            if not positions:  # Only enter if no positions are open
                # Buy signal: basket crosses above EMA
                if prev_basket <= prev_ema and curr_basket > curr_ema:
                    risk_amount = balance * (self.risk_per_trade / 100)
                    
                    for pair in self.pairs:
                        entry_price = data_dict[pair]['close'][i]
                        positions.append({
                            'type': 'buy',
                            'pair': pair,
                            'entry': entry_price,
                            'risk_amount': risk_amount / len(self.pairs)
                        })
                
                # Sell signal: basket crosses below EMA
                elif prev_basket >= prev_ema and curr_basket < curr_ema:
                    risk_amount = balance * (self.risk_per_trade / 100)
                    
                    for pair in self.pairs:
                        entry_price = data_dict[pair]['close'][i]
                        positions.append({
                            'type': 'sell',
                            'pair': pair,
                            'entry': entry_price,
                            'risk_amount': risk_amount / len(self.pairs)
                        })
            
            equity_curve.append(balance)
            dates.append(current_time)
        
        # Calculate performance metrics
        returns = pd.Series(equity_curve).pct_change().dropna()
        winning_trades = len([t for t in trades if t['profit'] > 0])
        total_trades = len(trades)
        
        stats = {
            'net_profit': balance - initial_balance,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'profit_factor': abs(sum(t['profit'] for t in trades if t['profit'] > 0)) / abs(sum(t['profit'] for t in trades if t['profit'] < 0)) if sum(t['profit'] for t in trades if t['profit'] < 0) != 0 else float('inf'),
            'total_trades': total_trades,
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() != 0 else 0,
            'avg_trade': sum(t['profit'] for t in trades) / total_trades if total_trades > 0 else 0,
            'recovery_factor': (balance - initial_balance) / (max(equity_curve) - min(equity_curve)) if max(equity_curve) - min(equity_curve) != 0 else 0
        }
        
        # Calculate monthly returns
        equity_series = pd.Series(equity_curve, index=dates)
        monthly_returns = equity_series.resample('M').last().pct_change().dropna()
        monthly_returns_list = [{'month': date.strftime('%Y-%m'), 'return': ret * 100} 
                              for date, ret in monthly_returns.items()]
        
        return {
            'stats': stats,
            'trades': trades,
            'dates': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
            'equity_curve': equity_curve,
            'basket_index': self.basket_data.tolist(),
            'ema_line': self.ema_line.tolist(),
            'monthly_returns': monthly_returns_list
        }
    
    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd 
from flask import Flask, render_template, request, jsonify
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import os
from werkzeug.utils import secure_filename
import subprocess
import json
import tempfile
import threading
import time
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
import requests
from dotenv import load_dotenv
import traceback
from config import SERVER_CONFIG, MT5_CONFIG
from strategies.basket_index_strategy import ForexBasketIndexStrategy

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MT5 credentials - Replace with your credentials
MT5_CONFIG = {
    "server": "Exness-MT5Trial7",
    "login": 203405414,
    "password": "Arun@123"
}

# Configuration for file uploads
UPLOAD_FOLDER = 'uploaded_eas'
ALLOWED_EXTENSIONS = {'ex4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MT4_PATH = "C:/Program Files/MetaTrader 4"  # Update this to your MT4 installation path

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for live trading
live_trading_active = False
live_trading_params = None
last_signal = None
active_positions = {}

# Add new configuration variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize API clients
alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY)
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Global variable to track EA status
ea_status = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def connect_mt5():
    """Initialize and test MT5 connection"""
    try:
        # Shutdown existing connections
        mt5.shutdown()
        
        # Initialize MT5
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"Failed to initialize MT5: {error}")
            return False, f"Failed to initialize MT5: {error}"
        
        # Attempt to login
        if not mt5.login(login=MT5_CONFIG["login"], 
                        password=MT5_CONFIG["password"], 
                        server=MT5_CONFIG["server"]):
            error = mt5.last_error()
            print(f"Failed to login to MT5: {error}")
            return False, f"Failed to login to MT5: {error}"
            
        # Get account info to verify connection
        account_info = mt5.account_info()
        if account_info is None:
            error = mt5.last_error()
            print(f"Failed to get account info: {error}")
            return False, f"Failed to get account info: {error}"
            
        print(f"Successfully connected to MT5 account {account_info.login}")
        print(f"Balance: {account_info.balance}")
        print(f"Equity: {account_info.equity}")
        return True, "Connected successfully"
        
    except Exception as e:
        print(f"Exception in connect_mt5: {str(e)}")
        return False, str(e)

def calculate_rsi(prices, period=14):
    """Calculate RSI for a price series"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_account_data():
    """Get current account information and positions"""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            return None

        # Get open positions
        positions = mt5.positions_get()
        open_positions = []
        if positions:
            for pos in positions:
                open_positions.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "Buy" if pos.type == 0 else "Sell",
                    "volume": pos.volume,
                    "open_price": pos.price_open,
                    "current_price": pos.price_current,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "sl": pos.sl,
                    "tp": pos.tp
                })

        # Get all closed trades (history)
        from_date = datetime(2000, 1, 1)  # Start from a very old date to get all history
        to_date = datetime.now()
        deals = mt5.history_deals_get(from_date, to_date)
        closed_orders = []
        if deals:
            for deal in deals:
                closed_orders.append({
                    "ticket": deal.ticket,
                    "symbol": deal.symbol,
                    "type": "Buy" if deal.type == 0 else "Sell",
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "commission": deal.commission,
                    "swap": deal.swap,
                    "time": datetime.fromtimestamp(deal.time)
                })
            
            # Sort by time, most recent first
            closed_orders.sort(key=lambda x: x["time"], reverse=True)

        return {
            "account": {
                "login": account_info.login,
                "server": account_info.server,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "margin_level": account_info.margin_level
            },
            "positions": open_positions,
            "history": closed_orders,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Error getting account data: {e}")
        return None

@app.route('/')
def index():
    """Main dashboard page"""
    data = get_account_data()
    if data is None:
        return "Error: Could not connect to MT5"
    return render_template('dashboard.html', data=data)

@app.route('/get_symbols')
def get_symbols():
    """Get available trading symbols"""
    try:
        # Get all symbols from MT5
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return jsonify({'error': 'Failed to get symbols'}), 500
            
        # Filter for forex pairs and format them
        available_pairs = []
        for symbol in all_symbols:
            # Include both standard forex pairs (6 chars) and pairs with 'm' suffix
            if (len(symbol.name) == 6 or (len(symbol.name) == 7 and symbol.name.endswith('m'))) and symbol.trade_mode != 0:
                available_pairs.append(symbol.name)
        
        # Sort the pairs alphabetically
        available_pairs.sort()
        
        return jsonify(available_pairs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_correlation', methods=['POST'])
def analyze_correlation():
    """Analyze correlation between two symbols"""
    try:
        print("Received correlation analysis request")
        data = request.get_json()
        print(f"Request data: {data}")
        
        # Extract parameters
        pair1 = data.get('pair1')
        pair2 = data.get('pair2')
        timeframe = data.get('timeframe', 'H4')
        bars = int(data.get('analysis_period', 30)) * 24
        rsi_period = int(data.get('rsi_period', 14))
        rsi_overbought = float(data.get('rsi_overbought', 70))
        rsi_oversold = float(data.get('rsi_oversold', 30))
        corr_entry_threshold = float(data.get('corr_entry_threshold', 0.7))
        corr_exit_threshold = float(data.get('corr_exit_threshold', 0.3))

        print(f"Processing pairs: {pair1} and {pair2}")
        print(f"Timeframe: {timeframe}, Bars: {bars}")

        # Map timeframe
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H4)

        # Get price data
        rates1 = mt5.copy_rates_from_pos(pair1, mt5_timeframe, 0, bars)
        rates2 = mt5.copy_rates_from_pos(pair2, mt5_timeframe, 0, bars)
        
        print(f"Retrieved data - Pair1: {rates1 is not None}, Pair2: {rates2 is not None}")
        
        if rates1 is None or rates2 is None:
            error_msg = f"Failed to fetch price data for {pair1 if rates1 is None else pair2}"
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 400
            
        # Convert to DataFrames
        df1 = pd.DataFrame(rates1)
        df2 = pd.DataFrame(rates2)
        
        print(f"DataFrame shapes - Pair1: {df1.shape}, Pair2: {df2.shape}")
        
        # Calculate returns
        df1['returns'] = df1['close'].pct_change()
        df2['returns'] = df2['close'].pct_change()
        
        # Calculate RSI
        df1['rsi'] = calculate_rsi(df1['close'], rsi_period)
        df2['rsi'] = calculate_rsi(df2['close'], rsi_period)
        
        # Calculate correlations
        price_corr = df1['close'].corr(df2['close'])
        rsi_corr = df1['rsi'].corr(df2['rsi'])
        returns_corr = df1['returns'].corr(df2['returns'])
        
        print(f"Correlations - Price: {price_corr}, RSI: {rsi_corr}, Returns: {returns_corr}")
        
        # Calculate rolling correlation
        rolling_corr = df1['close'].rolling(window=20).corr(df2['close'])
        
        # Calculate additional statistics
        stats1 = {
            'mean': float(df1['close'].mean()),
            'std': float(df1['close'].std()),
            'min': float(df1['close'].min()),
            'max': float(df1['close'].max()),
            'volatility': float(df1['returns'].std() * np.sqrt(252)),  # Annualized volatility
            'skewness': float(df1['returns'].skew()),
            'kurtosis': float(df1['returns'].kurtosis()),
            'sharpe': float((df1['returns'].mean() / df1['returns'].std()) * np.sqrt(252)) if df1['returns'].std() != 0 else 0,
            'current_price': float(df1['close'].iloc[-1]),
            'price_change': float((df1['close'].iloc[-1] / df1['close'].iloc[0] - 1) * 100)  # Percentage change
        }
        
        stats2 = {
            'mean': float(df2['close'].mean()),
            'std': float(df2['close'].std()),
            'min': float(df2['close'].min()),
            'max': float(df2['close'].max()),
            'volatility': float(df2['returns'].std() * np.sqrt(252)),  # Annualized volatility
            'skewness': float(df2['returns'].skew()),
            'kurtosis': float(df2['returns'].kurtosis()),
            'sharpe': float((df2['returns'].mean() / df2['returns'].std()) * np.sqrt(252)) if df2['returns'].std() != 0 else 0,
            'current_price': float(df2['close'].iloc[-1]),
            'price_change': float((df2['close'].iloc[-1] / df2['close'].iloc[0] - 1) * 100)  # Percentage change
        }

        # Analyze correlation signals
        signals = []
        in_trade = False
        current_signal = None
        
        for i, corr in enumerate(rolling_corr):
            if pd.isna(corr):
                continue
                
            if not in_trade and abs(corr) >= corr_entry_threshold:
                current_signal = {
                    'start_index': i,
                    'start_corr': float(corr) if not pd.isna(corr) else None
                }
                in_trade = True
            elif in_trade and abs(corr) <= corr_exit_threshold:
                current_signal['end_index'] = i
                current_signal['end_corr'] = float(corr) if not pd.isna(corr) else None
                current_signal['duration'] = i - current_signal['start_index']
                signals.append(current_signal)
                current_signal = None
                in_trade = False

        # Calculate statistics, handling NaN values
        signal_durations = [s['duration'] for s in signals]
        avg_duration = float(np.mean(signal_durations)) if signal_durations else 0
        max_duration = float(np.max(signal_durations)) if signal_durations else 0

        print("Preparing response data")
        response = {
            'timestamps': [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M') for x in df1['time']],
            'prices1': [float(x) if not pd.isna(x) else None for x in df1['close']],
            'prices2': [float(x) if not pd.isna(x) else None for x in df2['close']],
            'rsi1': [float(x) if not pd.isna(x) else None for x in df1['rsi']],
            'rsi2': [float(x) if not pd.isna(x) else None for x in df2['rsi']],
            'rolling_correlation': [float(x) if not pd.isna(x) else None for x in rolling_corr],
            'price_correlation': float(price_corr) if not pd.isna(price_corr) else None,
            'rsi_correlation': float(rsi_corr) if not pd.isna(rsi_corr) else None,
            'returns_correlation': float(returns_corr) if not pd.isna(returns_corr) else None,
            'signals': signals,
            'stats': {
                'total_signals': len(signals),
                'avg_duration': float(avg_duration) if not pd.isna(avg_duration) else 0,
                'max_duration': float(max_duration) if not pd.isna(max_duration) else 0,
                'current_correlation': float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 and not pd.isna(rolling_corr.iloc[-1]) else None
            },
            'pair1_stats': stats1,
            'pair2_stats': stats2
        }
        
        print("Sending response")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze_correlation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/place_trade', methods=['POST'])
def place_trade():
    """Place a new trade"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        trade_type = data.get('type')  # 'buy' or 'sell'
        volume = float(data.get('volume', 0.01))
        sl = float(data.get('sl', 0))
        tp = float(data.get('tp', 0))
        
        # Get current price
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return jsonify({'error': f'Symbol {symbol} not found'}), 400
            
        price = symbol_info.ask if trade_type.lower() == 'buy' else symbol_info.bid
        
        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if trade_type.lower() == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "python trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send trade
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return jsonify({'error': f'Trade failed: {result.comment}'}), 400
            
        return jsonify({
            'success': True,
            'order': {
                'ticket': result.order,
                'volume': volume,
                'price': price
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/close_trade', methods=['POST'])
def close_trade():
    """Close an existing trade"""
    try:
        data = request.get_json()
        ticket = data.get('ticket')
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return jsonify({'error': 'Position not found'}), 400
            
        position = position[0]
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "python close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send close request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return jsonify({'error': f'Close failed: {result.comment}'}), 400
            
        return jsonify({
            'success': True,
            'closed_position': {
                'ticket': ticket,
                'profit': position.profit
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_account')
def update_account():
    """Get latest account information"""
    data = get_account_data()
    if data is None:
        return jsonify({'error': 'Failed to get account data'}), 500
    return jsonify(data)

@app.route('/correlation')
def correlation():
    """Correlation analysis page"""
    return render_template('correlation.html')

@app.route('/strength')
def strength():
    """Currency Strength Meter page"""
    return render_template('strength.html')

@app.route('/pair_movements')
def pair_movements():
    """Pair Movements Analysis page"""
    return render_template('pair_movements.html')

@app.route('/market_movements')
def market_movements():
    """Market Movements page"""
    return render_template('market_movements.html')

@app.route('/api/all_symbols', methods=['GET'])
def get_all_symbols():
    """Get all available currency pairs and their current data"""
    try:
        # Get all symbols
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return jsonify({"error": "Failed to get symbols", "details": mt5.last_error()}), 400

        # Filter for forex pairs with 'm' suffix
        forex_pairs = []
        for symbol in all_symbols:
            # Check if it's a forex pair (ends with 'm' and has proper length)
            if symbol.name.endswith('m') and len(symbol.name) == 7:
                # Get the last tick
                tick = mt5.symbol_info_tick(symbol.name)
                if tick is not None:
                    forex_pairs.append({
                        "symbol": symbol.name,
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "spread": symbol.spread,
                        "digits": symbol.digits,
                        "description": symbol.description
                    })

        # Sort pairs alphabetically
        forex_pairs.sort(key=lambda x: x["symbol"])

        return jsonify({
            "status": "success",
            "count": len(forex_pairs),
            "symbols": [pair["symbol"] for pair in forex_pairs]  # Send just the symbol names for the dropdown
        })

    except Exception as e:
        print(f"Error getting symbols: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/currency_strength')
def get_currency_strength():
    """Calculate and return currency strength for all major currencies"""
    try:
        # Get timeframe from query parameter, default to '1D'
        timeframe = request.args.get('timeframe', '1D')
        
        # Map timeframe to MT5 timeframe
        tf_map = {
            '1D': mt5.TIMEFRAME_D1,
            '4H': mt5.TIMEFRAME_H4
        }
        mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_D1)
        
        currencies = [
            'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF',
            'SEK', 'NOK', 'SGD', 'HKD', 'TRY', 'ZAR', 'MXN', 'CNY', 'INR', 'BRL'
        ]
        
        # Define all possible pairs with 'm' suffix
        pairs = [
            # Major pairs
            'EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'NZDUSDm', 'USDCADm', 'USDCHFm',
            'EURGBPm', 'EURJPYm', 'EURAUDm', 'EURNZDm', 'EURCADm', 'EURCHFm',
            'GBPJPYm', 'GBPAUDm', 'GBPNZDm', 'GBPCADm', 'GBPCHFm',
            'AUDJPYm', 'NZDJPYm', 'CADJPYm', 'CHFJPYm',
            'AUDNZDm', 'AUDCADm', 'AUDCHFm',
            'NZDCADm', 'NZDCHFm',
            'CADCHFm',
            
            # Additional pairs
            'USDSEKm', 'USDNOKm', 'USDSGDm', 'USDHKDm', 'USDTRYm', 'USDZARm',
            'USDMXNm', 'USDCNHm', 'USDINRm', 'USDBRLm',
            'EURSEKm', 'EURNOKm', 'EURTRYm', 'EURZARm', 'EURMXNm',
            'GBPSEKm', 'GBPNOKm', 'GBPTRYm', 'GBPZARm',
            'CHFTRYm', 'CHFNOKm', 'CHFSEKm', 'JPYTRYm'
        ]

        # Initialize currency scores
        currency_scores = {currency: 0.0 for currency in currencies}
        valid_pairs_count = {currency: 0 for currency in currencies}

        # Calculate strength based on current prices
        for pair in pairs:
            tick = mt5.symbol_info_tick(pair)
            if tick is not None:
                base = pair[:3]
                quote = pair[3:6]  # Exclude the 'm' suffix
                
                # Get current and previous prices based on timeframe
                rates = mt5.copy_rates_from_pos(pair, mt5_timeframe, 0, 2)
                if rates is not None and len(rates) >= 2:
                    prev_close = rates[0]['close']
                    curr_close = rates[1]['close']
                    
                    # Calculate price change percentage
                    change = ((curr_close - prev_close) / prev_close) * 100
                    
                    # Update scores
                    if base in currency_scores:
                        currency_scores[base] += change
                        valid_pairs_count[base] += 1
                    if quote in currency_scores:
                        currency_scores[quote] -= change
                        valid_pairs_count[quote] += 1

        # Average the scores by the number of pairs for each currency
        for currency in currencies:
            if valid_pairs_count[currency] > 0:
                currency_scores[currency] /= valid_pairs_count[currency]

        # Normalize scores to 0-100 range
        min_score = min(currency_scores.values())
        max_score = max(currency_scores.values())
        score_range = max_score - min_score if max_score != min_score else 1

        normalized_scores = {}
        for currency in currencies:
            normalized_scores[currency] = ((currency_scores[currency] - min_score) / score_range) * 100

        # Create response with current timestamp and timeframe
        strength_data = [{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframe': timeframe,
            **normalized_scores
        }]

        return jsonify(strength_data)
        
    except Exception as e:
        print(f"Error calculating currency strength: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/strongest_currencies')
def get_strongest_currencies():
    """Get the strongest currencies with their details"""
    try:
        # Get all symbols with 'm' suffix
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return jsonify({"error": "Failed to get symbols"}), 500

        # Get currency strength data
        strength_data = get_currency_strength().get_json()
        if not strength_data or 'error' in strength_data:
            return jsonify({"error": "Failed to get currency strength"}), 500

        latest_strength = strength_data[0]  # Get the latest strength data
        
        # Get the strongest currencies sorted by strength
        currencies = [
            'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF',
            'SEK', 'NOK', 'SGD', 'HKD', 'TRY', 'ZAR', 'MXN', 'CNY', 'INR', 'BRL'
        ]
        
        currency_data = []
        for currency in currencies:
            # Find a representative pair for this currency (preferably against USD)
            base_pair = None
            if currency != 'USD':
                base_pair = f"{currency}USDm"
            else:
                base_pair = "EURUSDm"  # Use EURUSD for USD strength
            
            # Get the latest tick and daily change
            if base_pair:
                tick = mt5.symbol_info_tick(base_pair)
                rates = mt5.copy_rates_from_pos(base_pair, mt5.TIMEFRAME_D1, 0, 2)
                
                if tick is not None and rates is not None and len(rates) >= 2:
                    prev_close = rates[0]['close']
                    curr_close = rates[1]['close']
                    
                    # Calculate change
                    change_pct = ((curr_close - prev_close) / prev_close) * 100
                    change_pips = (curr_close - prev_close) * (10000 if 'JPY' not in base_pair else 100)
                    
                    # Adjust for quote currency
                    if currency == 'USD':
                        change_pct *= -1
                        change_pips *= -1
                    
                    currency_data.append({
                        'currency': currency,
                        'strength': latest_strength[currency],
                        'change_percent': change_pct,
                        'change_pips': change_pips,
                        'last_price': curr_close
                    })
        
        # Sort by strength in descending order
        currency_data.sort(key=lambda x: x['strength'], reverse=True)
        
        return jsonify({
            'timestamp': latest_strength['timestamp'],
            'currencies': currency_data
        })
        
    except Exception as e:
        print(f"Error getting strongest currencies: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/strongest_pairs')
def get_strongest_pairs():
    """Get the strongest currency pairs with their details"""
    try:
        print("Fetching strongest pairs...")
        # Get all symbols with 'm' suffix
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            print("Failed to get symbols from MT5")
            return jsonify({"error": "Failed to get symbols"}), 500

        pairs_data = []
        for symbol in all_symbols:
            # Only include forex pairs with 'm' suffix
            if (len(symbol.name) == 7 and symbol.name.endswith('m')) and symbol.trade_mode != 0:
                print(f"Processing pair: {symbol.name}")
                # Get the latest tick and daily change
                tick = mt5.symbol_info_tick(symbol.name)
                rates = mt5.copy_rates_from_pos(symbol.name, mt5.TIMEFRAME_D1, 0, 2)
                
                if tick is not None and rates is not None and len(rates) >= 2:
                    prev_close = rates[0]['close']
                    curr_close = rates[1]['close']
                    
                    # Calculate changes
                    change_pct = ((curr_close - prev_close) / prev_close) * 100
                    change_pips = (curr_close - prev_close) * (10000 if 'JPY' not in symbol.name else 100)
                    
                    pairs_data.append({
                        'symbol': symbol.name[:-1],  # Remove 'm' suffix
                        'change_percent': change_pct,
                        'change_pips': change_pips,
                        'last_price': curr_close,
                        'spread': symbol.spread,
                        'volume': int(rates[1]['tick_volume'])  # Convert uint64 to int
                    })
                    print(f"Added pair {symbol.name[:-1]} with change {change_pct:.2f}%")
        
        if not pairs_data:
            print("No pairs data collected")
            return jsonify({"error": "No pairs data available"}), 500
            
        # Sort by absolute percentage change in descending order
        pairs_data.sort(key=lambda x: abs(x['change_percent']), reverse=True)
        
        response_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pairs': pairs_data[:20]  # Return top 20 most volatile pairs
        }
        print(f"Returning {len(response_data['pairs'])} pairs")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error getting strongest pairs: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_movements')
def get_market_movements():
    """Get market movements data including volatility and sentiment"""
    try:
        # Get timeframe from query parameter, default to '1H'
        timeframe = request.args.get('timeframe', '1H')
        
        # Map timeframe to MT5 timeframe and number of candles
        tf_map = {
            '1H': (mt5.TIMEFRAME_H1, 1),
            '2H': (mt5.TIMEFRAME_H1, 2),  # Use 2 1-hour candles for 2H
            '1D': (mt5.TIMEFRAME_D1, 1)
        }
        mt5_timeframe, num_candles = tf_map.get(timeframe, (mt5.TIMEFRAME_H1, 1))
        
        # Get major pairs data
        major_pairs = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCHFm', 'USDCADm', 'AUDUSDm', 'NZDUSDm']
        major_pairs_data = []
        
        for pair in major_pairs:
            rates = mt5.copy_rates_from_pos(pair, mt5_timeframe, 0, num_candles + 1)
            if rates is not None and len(rates) >= num_candles + 1:
                prev_close = rates[0]['close']
                curr_close = rates[-1]['close']
                change = ((curr_close - prev_close) / prev_close) * 100
                major_pairs_data.append({
                    'symbol': pair[:-1],
                    'change': change
                })
        
        # Get volatility data (24-hour)
        volatility_data = {
            'timestamps': [],
            'values': []
        }
        
        for pair in major_pairs:
            rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_H1, 0, 24)
            if rates is not None:
                for i, rate in enumerate(rates):
                    if i == 0:
                        continue
                    timestamp = datetime.fromtimestamp(rate['time']).strftime('%H:%M')
                    volatility = abs((rate['high'] - rate['low']) / rate['low'] * 100)
                    
                    if i >= len(volatility_data['timestamps']):
                        volatility_data['timestamps'].append(timestamp)
                        volatility_data['values'].append(volatility)
                    else:
                        volatility_data['values'][i-1] += volatility
        
        # Average volatility across pairs
        volatility_data['values'] = [v / len(major_pairs) for v in volatility_data['values']]
        
        # Calculate market sentiment
        sentiment_data = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        }
        
        all_symbols = mt5.symbols_get()
        if all_symbols is not None:
            for symbol in all_symbols:
                if (len(symbol.name) == 7 and symbol.name.endswith('m')) and symbol.trade_mode != 0:
                    rates = mt5.copy_rates_from_pos(symbol.name, mt5_timeframe, 0, num_candles + 1)
                    if rates is not None and len(rates) >= num_candles + 1:
                        change = ((rates[-1]['close'] - rates[0]['close']) / rates[0]['close']) * 100
                        if change > 0.1:
                            sentiment_data['bullish'] += 1
                        elif change < -0.1:
                            sentiment_data['bearish'] += 1
                        else:
                            sentiment_data['neutral'] += 1
        
        # Get detailed movement analysis
        movements = []
        for symbol in all_symbols:
            if (len(symbol.name) == 7 and symbol.name.endswith('m')) and symbol.trade_mode != 0:
                rates = mt5.copy_rates_from_pos(symbol.name, mt5_timeframe, 0, 20)
                if rates is not None and len(rates) >= num_candles + 1:
                    curr_close = rates[-1]['close']
                    prev_close = rates[0]['close']
                    change = ((curr_close - prev_close) / prev_close) * 100
                    pips = (curr_close - prev_close) * (10000 if 'JPY' not in symbol.name else 100)
                    
                    # Calculate volatility
                    volatility = np.std([((r['high'] - r['low']) / r['low']) * 100 for r in rates])
                    
                    # Determine trend
                    closes = [r['close'] for r in rates]
                    ma_short = np.mean(closes[-5:])
                    ma_long = np.mean(closes[-20:])
                    trend = 'up' if ma_short > ma_long else 'down' if ma_short < ma_long else 'neutral'
                    
                    movements.append({
                        'symbol': symbol.name[:-1],
                        'change': change,
                        'pips': abs(pips),
                        'volatility': volatility,
                        'trend': trend
                    })
        
        # Sort movements by absolute change
        movements.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return jsonify({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframe': timeframe,
            'major_pairs': major_pairs_data,
            'volatility': volatility_data,
            'sentiment': sentiment_data,
            'movements': movements[:30]  # Return top 30 movements
        })
        
    except Exception as e:
        print(f"Error getting market movements: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/account_info')
def get_account_info():
    """Get current account information and positions"""
    try:
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            return jsonify({'error': 'Failed to get account info'}), 500

        # Get open positions
        positions = mt5.positions_get()
        positions_list = []
        
        if positions is not None:
            for pos in positions:
                positions_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'Buy' if pos.type == 0 else 'Sell',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'time': datetime.fromtimestamp(pos.time).strftime('%Y-%m-%d %H:%M:%S')
                })

        # Get closed trades (last 24 hours by default)
        from_date = datetime.now() - timedelta(days=1)
        closed_trades = mt5.history_deals_get(from_date)
        closed_trades_list = []

        if closed_trades is not None:
            for trade in closed_trades:
                if trade.type <= 1:  # Only include buy and sell trades
                    closed_trades_list.append({
                        'ticket': trade.ticket,
                        'symbol': trade.symbol,
                        'type': 'Buy' if trade.type == 0 else 'Sell',
                        'volume': trade.volume,
                        'price': trade.price,
                        'profit': trade.profit,
                        'time': datetime.fromtimestamp(trade.time).strftime('%Y-%m-%d %H:%M:%S')
                    })

        return jsonify({
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin_free': account_info.margin_free,
            'floating_pl': account_info.equity - account_info.balance,
            'positions': positions_list,
            'closed_trades': closed_trades_list
        })

    except Exception as e:
        print(f"Error in get_account_info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strength_data')
def get_strength_data():
    try:
        currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        currency_data = []
        
        for currency in currencies:
            # Calculate strength based on multiple pairs
            pairs = []
            strength = 0
            total_weight = 0
            
            # Define pairs for each currency
            if currency == 'USD':
                pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCHF', 'USDJPY', 'USDCAD']
            elif currency == 'EUR':
                pairs = ['EURUSD', 'EURGBP', 'EURAUD', 'EURNZD', 'EURCHF', 'EURJPY', 'EURCAD']
            elif currency == 'GBP':
                pairs = ['GBPUSD', 'EURGBP', 'GBPAUD', 'GBPNZD', 'GBPCHF', 'GBPJPY', 'GBPCAD']
            else:
                # For other currencies, create pairs with major currencies
                for major in ['USD', 'EUR', 'GBP']:
                    if major != currency:
                        pairs.append(f"{currency}{major}")
                        pairs.append(f"{major}{currency}")
            
            # Calculate weighted strength
            for pair in pairs:
                try:
                    rates = mt5.copy_rates_from_pos(pair + 'm', mt5.TIMEFRAME_H1, 0, 2)
                    if rates is not None and len(rates) >= 2:
                        prev_close = rates[0]['close']
                        curr_close = rates[1]['close']
                        
                        # Calculate change percentage
                        change = ((curr_close - prev_close) / prev_close) * 100
                        
                        # Adjust change based on position in pair
                        if pair.startswith(currency):
                            strength += change
                        else:
                            strength -= change
                            
                        total_weight += 1
                except:
                    continue
            
            # Calculate average strength
            avg_strength = strength / total_weight if total_weight > 0 else 0
            
            # Normalize to 0-1 range (will be converted to percentage in frontend)
            normalized_strength = (avg_strength + 5) / 10  # Assuming typical range is -5% to +5%
            normalized_strength = max(0, min(1, normalized_strength))  # Clamp between 0 and 1
            
            currency_data.append({
                'code': currency,
                'strength': round(normalized_strength, 3),
                'change': round(avg_strength, 3)
            })
        
        # Sort by strength
        currency_data.sort(key=lambda x: x['strength'], reverse=True)
        
        # Calculate opportunities
        opportunities = []
        for i in range(len(currency_data)):
            for j in range(i + 1, len(currency_data)):
                strong = currency_data[i]
                weak = currency_data[j]
                strength_diff = strong['strength'] - weak['strength']
                
                if strength_diff > 0.3:  # Only show significant differences
                    opportunities.append({
                        'pair': f"{strong['code']}/{weak['code']}",
                        'type': 'Buy',
                        'strength_diff': round(strength_diff * 100, 1)
                    })
                elif strength_diff < -0.3:
                    opportunities.append({
                        'pair': f"{weak['code']}/{strong['code']}",
                        'type': 'Sell',
                        'strength_diff': round(abs(strength_diff) * 100, 1)
                    })
        
        return jsonify({
            'currencies': currency_data,
            'opportunities': sorted(opportunities, key=lambda x: x['strength_diff'], reverse=True)[:5],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"Error in get_strength_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pair_movements')
def get_pair_movements():
    """Get all pairs that moved more than 1% in last 2 hours with their graph data"""
    try:
        # Get all symbols with 'm' suffix
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return jsonify({"error": "Failed to get symbols"}), 500

        pairs_data = []
        for symbol in all_symbols:
            # Only include forex pairs with 'm' suffix
            if (len(symbol.name) == 7 and symbol.name.endswith('m')) and symbol.trade_mode != 0:
                # Get 2 hours of data with 5-minute intervals (24 points)
                rates = mt5.copy_rates_from_pos(symbol.name, mt5.TIMEFRAME_M5, 0, 24)
                
                if rates is not None and len(rates) >= 24:
                    # Calculate percentage change from 2 hours ago
                    start_price = rates[0]['close']
                    current_price = rates[-1]['close']
                    change_pct = ((current_price - start_price) / start_price) * 100
                    
                    # Only include pairs that moved more than 1%
                    if abs(change_pct) >= 1:
                        # Prepare graph data
                        graph_data = {
                            'timestamps': [datetime.fromtimestamp(rate['time']).strftime('%H:%M') for rate in rates],
                            'prices': [rate['close'] for rate in rates]
                        }
                        
                        pairs_data.append({
                            'symbol': symbol.name[:-1],  # Remove 'm' suffix
                            'change_percent': change_pct,
                            'start_price': start_price,
                            'current_price': current_price,
                            'graph_data': graph_data
                        })
        
        # Sort by absolute percentage change in descending order
        pairs_data.sort(key=lambda x: abs(x['change_percent']), reverse=True)
        
        return jsonify({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pairs': pairs_data
        })
        
    except Exception as e:
        print(f"Error getting pair movements: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/closed_trades')
def get_closed_trades():
    """Get all closed trades history"""
    try:
        print("Starting to fetch closed trades...")
        
        # First ensure we have a valid MT5 connection
        if not mt5.initialize():
            return jsonify({'error': 'Failed to initialize MT5 connection', 'trades': []})
            
        # Reconnect to the account if needed
        if not mt5.login(login=MT5_CONFIG["login"], 
                        password=MT5_CONFIG["password"], 
                        server=MT5_CONFIG["server"]):
            return jsonify({'error': 'Failed to login to MT5', 'trades': []})
        
        # Use a very old start date to get all history
        start_time = datetime(2000, 1, 1)
        end_time = datetime.now()
        
        print(f"Fetching trades from {start_time} to {end_time}")
        
        # Get all deals from MT5
        deals = mt5.history_deals_get(start_time, end_time)
        
        if deals is None:
            error = mt5.last_error()
            error_msg = f"Failed to get trades: {error}" if error else "No trades found"
            print(error_msg)
            return jsonify({'error': error_msg, 'trades': []})
        
        print(f"Found {len(deals)} total deals")
        
        # Process and format trades
        processed_trades = []
        positions_dict = {}  # To store position info
        
        # First pass: collect all positions
        for deal in deals:
            if deal.position_id not in positions_dict:
                positions_dict[deal.position_id] = {
                    'open_time': deal.time,
                    'open_price': deal.price,
                    'close_time': deal.time,
                    'close_price': deal.price,
                    'profit': deal.profit,
                    'volume': deal.volume,
                    'symbol': deal.symbol,
                    'type': deal.type,
                    'commission': deal.commission,
                    'swap': deal.swap
                }
            else:
                # Update position with closing details
                pos = positions_dict[deal.position_id]
                if deal.time > pos['close_time']:
                    pos['close_time'] = deal.time
                    pos['close_price'] = deal.price
                    pos['profit'] += deal.profit
                    pos['commission'] += deal.commission
                    pos['swap'] += deal.swap
        
        # Convert positions to trades
        for pos_id, pos in positions_dict.items():
            if pos['profit'] != 0:  # Only include completed trades
                processed_trades.append({
                    'close_time': int(pos['close_time'] * 1000),  # Convert to milliseconds for JS
                    'symbol': pos['symbol'],
                    'type': 'buy' if pos['type'] == 0 else 'sell',
                    'volume': float(pos['volume']),
                    'open_price': float(pos['open_price']),
                    'close_price': float(pos['close_price']),
                    'profit': float(pos['profit']),
                    'commission': float(pos['commission']),
                    'swap': float(pos['swap'])
                })
        
        print(f"Successfully processed {len(processed_trades)} valid trades")
        
        # Sort by close time, most recent first
        processed_trades.sort(key=lambda x: x['close_time'], reverse=True)
        
        return jsonify({
            'trades': processed_trades,
            'total_trades': len(processed_trades),
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"Error in get_closed_trades: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'trades': []}), 500

@app.route('/backtester')
def backtester():
    return render_template('backtester.html')

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.get_json()
        
        # Ensure MT5 is connected
        if not mt5.initialize():
            mt5.shutdown()
            if not mt5.initialize():
                raise Exception("Failed to connect to MetaTrader5")
        
        # Parse dates
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
        
        # Initialize strategy
        strategy = ForexBasketIndexStrategy(
            ema_period=data['ema_period'],
            risk_per_trade=data['risk_per_trade']
        )
        
        # Run backtest
        results = strategy.backtest(
            timeframe=data['timeframe'],
            start_date=start_date,
            end_date=end_date,
            initial_balance=data['initial_balance']
        )
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Backtest error: {str(e)}")  # Add logging
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_ea', methods=['POST'])
def analyze_ea():
    try:
        ea_file = None
        filepath = None
        
        # Check if file was uploaded in request
        if 'eaFile' in request.files:
            ea_file = request.files['eaFile']
            if ea_file.filename != '':
                if not ea_file.filename.endswith('.ex4'):
                    return jsonify({'error': 'Invalid file type. Please upload an .ex4 file'}), 400
                    
                # Save uploaded file
                filename = secure_filename(ea_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                ea_file.save(filepath)
        
        # If no file uploaded, check for existing file
        if filepath is None or not os.path.exists(filepath):
            # Look for EA file in root directory
            for file in os.listdir('.'):
                if file.endswith('.ex4'):
                    filepath = os.path.join('.', file)
                    break
        
        if filepath is None or not os.path.exists(filepath):
            return jsonify({'error': 'No EA file found'}), 400

        print(f"Analyzing EA file: {filepath}")
        
        # Get EA info from MT5
        ea_info = {
            'name': 'Basket Trading EA',
            'inputs': [
                {'name': 'LotSize', 'type': 'double', 'default': 0.01, 'description': 'Trading lot size'},
                {'name': 'RiskPercent', 'type': 'double', 'default': 2.0, 'description': 'Risk per trade (%)'},
                {'name': 'Magic', 'type': 'integer', 'default': 123456, 'description': 'Magic Number'}
            ]
        }
        
        return jsonify(ea_info)

    except Exception as e:
        print(f"Error analyzing EA: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/basket_backtest', methods=['POST'])
def basket_backtest():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        params = request.get_json()
        
        # Validate required parameters
        required_params = ['initialBalance', 'startDate', 'endDate', 'lotSize', 'timeframe', 'emaPeriod']
        for param in required_params:
            if param not in params:
                return jsonify({'error': f'Missing required parameter: {param}'}), 400
        
        # Parse dates (ensure they're in UTC)
        try:
            # Make sure we get proper datetime objects
            start_date = pd.to_datetime(params['startDate']).tz_localize(None)
            end_date = pd.to_datetime(params['endDate']).tz_localize(None)
            
            # Convert to timestamp integers that MT5 can use
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            print(f"Date range: {start_date} to {end_date}")
            print(f"Timestamp range: {start_timestamp} to {end_timestamp}")
        except Exception as e:
            return jsonify({'error': f'Invalid date format: {str(e)}'}), 400
            
        # Map timeframe to MT5 constant
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        mt5_timeframe = timeframe_map.get(params['timeframe'], mt5.TIMEFRAME_H1)
        print(f"Using timeframe: {params['timeframe']} (MT5: {mt5_timeframe})")
        
        # Verify MT5 connection
        if not mt5.initialize():
            error = mt5.last_error()
            return jsonify({'error': f'Failed to initialize MT5 connection: {error}'}), 500
        
        # Get all available symbols
        symbols = mt5.symbols_get()
        available_symbols = [symbol.name for symbol in symbols] if symbols else []
        print(f"Available symbols: {available_symbols[:20]}...")  # Print first 20 symbols to avoid flooding the log
        
        # Define pairs we want to use - check if they exist with or without 'm' suffix
        base_pairs = ['GBPUSD', 'EURUSD', 'NZDUSD', 'AUDUSD']
        pairs = []
        
        for pair in base_pairs:
            # First try with 'm' suffix
            if f"{pair}m" in available_symbols:
                pairs.append(f"{pair}m")
            # Then try without suffix
            elif pair in available_symbols:
                pairs.append(pair)
            else:
                return jsonify({'error': f'Symbol {pair} not found in MT5. Available symbols: {available_symbols[:10]}...'}), 400
        
        print(f"Using pairs: {pairs}")
        
        # Get historical data for all pairs
        pair_data = {}
        
        for pair in pairs:
            print(f"Fetching data for {pair} from {start_date} to {end_date}")
            
            # Try using copy_rates_range first
            rates = mt5.copy_rates_range(pair, mt5_timeframe, start_timestamp, end_timestamp)
            
            # If that fails, try using copy_rates_from_pos as fallback
            if rates is None:
                error_code = mt5.last_error()
                print(f"Failed to get data using copy_rates_range for {pair}. Error code: {error_code}")
                print("Trying alternative method...")
                
                # Try to get at least 1000 bars or up to the date range
                rates = mt5.copy_rates_from_pos(pair, mt5_timeframe, 0, 1000)
                
                if rates is None:
                    error_code = mt5.last_error()
                    return jsonify({'error': f'Failed to get data for {pair} using both methods. Error code: {error_code}'}), 400
                
                print(f"Successfully retrieved {len(rates) if rates is not None else 0} bars using alternative method")
            
            if len(rates) == 0:
                return jsonify({'error': f'No data available for {pair} in the selected period'}), 400
                
            pair_data[pair] = pd.DataFrame(rates)
            pair_data[pair]['time'] = pd.to_datetime(pair_data[pair]['time'], unit='s')
            print(f"Retrieved {len(pair_data[pair])} bars for {pair}")
        
        # Calculate basket index
        basket_df = pd.DataFrame()
        basket_df['time'] = pair_data[pairs[0]]['time']
        basket_df['basket_price'] = sum(pair_data[pair]['close'] for pair in pairs) / len(pairs)
        
        # Calculate EMA
        basket_df['ema'] = basket_df['basket_price'].ewm(span=params['emaPeriod'], adjust=False).mean()
        
        # Initialize results
        balance = params['initialBalance']
        equity = params['initialBalance']
        trades = []
        detailed_trades = []  # Will store detailed trade information
        equity_curve = [params['initialBalance']]
        dates = [basket_df['time'].iloc[0].strftime('%Y-%m-%d %H:%M')]
        
        # Trading variables
        current_direction = None  # None, 'buy', or 'sell'
        position_entry_prices = {}  # To track entry prices for each pair
        position_entry_time = None  # Track when the basket position was opened
        
        # Trading logic - EMA crossover
        for i in range(1, len(basket_df)):
            current_bar = basket_df.iloc[i]
            prev_bar = basket_df.iloc[i-1]
            
            # Check for EMA crossover
            cross_above_ema = prev_bar['basket_price'] <= prev_bar['ema'] and current_bar['basket_price'] > current_bar['ema']
            cross_below_ema = prev_bar['basket_price'] >= prev_bar['ema'] and current_bar['basket_price'] < current_bar['ema']
            
            # Process signals
            if cross_above_ema:  # Buy signal
                # Close any existing sell positions
                if current_direction == 'sell':
                    trade_profit = 0
                    pair_profits = {}  # Track profit for each pair
                    
                    for pair in pairs:
                        current_price = pair_data[pair].iloc[i]['close']
                        entry_price = position_entry_prices[pair]
                        # Calculate profit for sell position being closed
                        pair_profit = (entry_price - current_price) * params['lotSize'] * 100000
                        trade_profit += pair_profit
                        pair_profits[pair] = pair_profit
                        
                    # Record the trade with detailed information
                    close_time = current_bar['time'].strftime('%Y-%m-%d %H:%M')
                    trades.append({
                        'date': close_time,
                        'type': 'sell (closed)',
                        'entry': 'multiple',
                        'exit': 'multiple',
                        'profit': trade_profit,
                        'exit_reason': 'opposite signal'
                    })
                    
                    # Record detailed trade info
                    detailed_trades.append({
                        'basket_type': 'sell',
                        'entry_date': position_entry_time,
                        'exit_date': close_time,
                        'total_profit': trade_profit,
                        'pair_profits': pair_profits,
                        'exit_reason': 'Buy signal (crossover)'
                    })
                    
                    # Update balance
                    balance += trade_profit
                
                # Open buy positions
                position_entry_prices = {}
                position_entry_time = current_bar['time'].strftime('%Y-%m-%d %H:%M')
                
                for pair in pairs:
                    current_price = pair_data[pair].iloc[i]['close']
                    position_entry_prices[pair] = current_price
                
                current_direction = 'buy'
                
            elif cross_below_ema:  # Sell signal
                # Close any existing buy positions
                if current_direction == 'buy':
                    trade_profit = 0
                    pair_profits = {}  # Track profit for each pair
                    
                    for pair in pairs:
                        current_price = pair_data[pair].iloc[i]['close']
                        entry_price = position_entry_prices[pair]
                        # Calculate profit for buy position being closed
                        pair_profit = (current_price - entry_price) * params['lotSize'] * 100000
                        trade_profit += pair_profit
                        pair_profits[pair] = pair_profit
                        
                    # Record the trade with detailed information
                    close_time = current_bar['time'].strftime('%Y-%m-%d %H:%M')
                    trades.append({
                        'date': close_time,
                        'type': 'buy (closed)',
                        'entry': 'multiple',
                        'exit': 'multiple',
                        'profit': trade_profit,
                        'exit_reason': 'opposite signal'
                    })
                    
                    # Record detailed trade info
                    detailed_trades.append({
                        'basket_type': 'buy',
                        'entry_date': position_entry_time,
                        'exit_date': close_time,
                        'total_profit': trade_profit,
                        'pair_profits': pair_profits,
                        'exit_reason': 'Sell signal (crossover)'
                    })
                    
                    # Update balance
                    balance += trade_profit
                
                # Open sell positions
                position_entry_prices = {}
                position_entry_time = current_bar['time'].strftime('%Y-%m-%d %H:%M')
                
                for pair in pairs:
                    current_price = pair_data[pair].iloc[i]['close']
                    position_entry_prices[pair] = current_price
                
                current_direction = 'sell'
            
            # Update equity curve (daily)
            if i % 24 == 0:
                # Calculate unrealized P/L
                unrealized_profit = 0
                if current_direction and position_entry_prices:
                    for pair in pairs:
                        current_price = pair_data[pair].iloc[i]['close']
                        entry_price = position_entry_prices[pair]
                        
                        if current_direction == 'buy':
                            unrealized_profit += (current_price - entry_price) * params['lotSize'] * 100000
                        else:  # sell
                            unrealized_profit += (entry_price - current_price) * params['lotSize'] * 100000
                
                equity = balance + unrealized_profit
                equity_curve.append(equity)
                dates.append(current_bar['time'].strftime('%Y-%m-%d %H:%M'))
        
        # Close any remaining positions at the end of the backtest
        if current_direction:
            trade_profit = 0
            pair_profits = {}  # Track profit for each pair
            last_bar = basket_df.iloc[-1]
            
            for pair in pairs:
                current_price = pair_data[pair].iloc[-1]['close']
                entry_price = position_entry_prices[pair]
                
                if current_direction == 'buy':
                    pair_profit = (current_price - entry_price) * params['lotSize'] * 100000
                else:  # sell
                    pair_profit = (entry_price - current_price) * params['lotSize'] * 100000
                    
                trade_profit += pair_profit
                pair_profits[pair] = pair_profit
            
            # Record the final trade
            close_time = last_bar['time'].strftime('%Y-%m-%d %H:%M')
            trades.append({
                'date': close_time,
                'type': f'{current_direction} (closed)',
                'entry': 'multiple',
                'exit': 'multiple',
                'profit': trade_profit,
                'exit_reason': 'end of test'
            })
            
            # Record detailed trade info
            detailed_trades.append({
                'basket_type': current_direction,
                'entry_date': position_entry_time,
                'exit_date': close_time,
                'total_profit': trade_profit,
                'pair_profits': pair_profits,
                'exit_reason': 'End of backtest'
            })
            
            balance += trade_profit
        
        # Calculate statistics
        profitable_trades = len([t for t in trades if t['profit'] > 0])
        total_trades = len(trades)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        gross_profit = sum([t['profit'] for t in trades if t['profit'] > 0])
        gross_loss = abs(sum([t['profit'] for t in trades if t['profit'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        max_equity = max(equity_curve)
        min_equity = min(equity_curve)
        max_drawdown = ((max_equity - min_equity) / max_equity) * 100
        
        # Calculate average win/loss and largest win/loss
        wins = [t['profit'] for t in trades if t['profit'] > 0]
        losses = [t['profit'] for t in trades if t['profit'] < 0]
        
        average_win = sum(wins) / len(wins) if wins else 0
        average_loss = sum(losses) / len(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Calculate statistics for individual pairs
        pair_stats = {}
        for pair in pairs:
            pair_profits = [t['pair_profits'][pair] for t in detailed_trades]
            profitable_pair_trades = len([p for p in pair_profits if p > 0])
            pair_win_rate = (profitable_pair_trades / len(pair_profits) * 100) if pair_profits else 0
            pair_net_profit = sum(pair_profits)
            
            pair_stats[pair] = {
                'total_trades': len(pair_profits),
                'profitable_trades': profitable_pair_trades,
                'win_rate': pair_win_rate,
                'net_profit': pair_net_profit,
                'average_profit': sum(pair_profits) / len(pair_profits) if pair_profits else 0
            }
        
        print(f"Backtest completed: {total_trades} trades, Win Rate: {win_rate:.2f}%, Profit Factor: {profit_factor:.2f}")
        
        return jsonify({
            'dates': dates,
            'basketIndex': basket_df['basket_price'].tolist(),
            'ema': basket_df['ema'].tolist(),
            'equityCurve': equity_curve,
            'trades': trades,
            'detailed_trades': detailed_trades,
            'netProfit': balance - params['initialBalance'],
            'profitFactor': profit_factor,
            'winRate': win_rate,
            'maxDrawdown': max_drawdown,
            'totalTrades': total_trades,
            'profitableTrades': profitable_trades,
            'lossTrades': total_trades - profitable_trades,
            'averageWin': average_win,
            'averageLoss': average_loss,
            'largestWin': largest_win,
            'largestLoss': largest_loss,
            'initialDeposit': params['initialBalance'],
            'finalBalance': balance,
            'pair_stats': pair_stats,
            'parameters': params
        })
        
    except Exception as e:
        print(f"Error in basket backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_live_trading', methods=['POST'])
def start_live_trading():
    try:
        global live_trading_active, live_trading_params
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        
        # Validate and store parameters
        live_trading_params = {
            'tradingMode': data.get('tradingMode', 'auto'),
            'riskPerTrade': float(data.get('riskPerTrade', 1.0)),
            'timeframe': data.get('timeframe', 'H1'),
            'emaPeriod': int(data.get('emaPeriod', 200))
        }
        
        # Map timeframe
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        live_trading_params['mt5_timeframe'] = timeframe_map.get(live_trading_params['timeframe'], mt5.TIMEFRAME_H1)
        
        # Start the trading thread if not already running
        if not live_trading_active:
            live_trading_active = True
            threading.Thread(target=monitor_live_trading, daemon=True).start()
            
        return jsonify({'status': 'success', 'message': 'Live trading started'})
        
    except Exception as e:
        print(f"Error starting live trading: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_live_trading', methods=['POST'])
def stop_live_trading():
    try:
        global live_trading_active
        live_trading_active = False
        return jsonify({'status': 'success', 'message': 'Live trading stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live_trading_status')
def get_live_trading_status():
    try:
        global last_signal
        
        # Define base pairs
        base_pairs = ['GBPUSD', 'EURUSD', 'NZDUSD', 'AUDUSD']
        
        # Get available symbols
        symbols = mt5.symbols_get()
        available_symbols = [symbol.name for symbol in symbols] if symbols else []
        
        # Find correct symbol names
        pairs = []
        for pair in base_pairs:
            if f"{pair}m" in available_symbols:
                pairs.append(f"{pair}m")
            elif pair in available_symbols:
                pairs.append(pair)
        
        # Get active positions
        positions = []
        
        for pair in pairs:
            pair_positions = mt5.positions_get(symbol=pair)
            if pair_positions is not None:
                for position in pair_positions:
                    positions.append({
                        'symbol': position.symbol,
                        'type': 'Buy' if position.type == 0 else 'Sell',
                        'entry': position.price_open,
                        'current': position.price_current,
                        'sl': position.sl,
                        'tp': position.tp,
                        'profit': position.profit,
                        'ticket': position.ticket
                    })
        
        # Calculate basket value (average of current prices)
        basket_value = 0.0
        basket_values_count = 0
        
        for pair in pairs:
            ticker = mt5.symbol_info_tick(pair)
            if ticker is not None:
                basket_value += (ticker.bid + ticker.ask) / 2
                basket_values_count += 1
        
        if basket_values_count > 0:
            basket_value /= basket_values_count
        
        # Calculate signal strength (simplified)
        signal_strength = 0
        
        return jsonify({
            'status': 'active' if live_trading_active else 'inactive',
            'lastSignal': last_signal,
            'basketValue': basket_value,
            'signalStrength': signal_strength,
            'positions': positions
        })
        
    except Exception as e:
        print(f"Error getting live trading status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/close_position', methods=['POST'])
def close_position():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        
        if 'ticket' not in data:
            return jsonify({'error': 'Missing required parameter: ticket'}), 400
            
        # Ensure ticket is an integer
        ticket = int(data['ticket'])
        
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return jsonify({'error': f'Position with ticket {ticket} not found'}), 404
            
        position = position[0]
        
        # Create order request to close position
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,  # Opposite of position type
            "price": mt5.symbol_info_tick(position.symbol).ask if position.type == 1 else mt5.symbol_info_tick(position.symbol).bid,
            "deviation": 20,
            "magic": 123456,
            "comment": "Close position from web interface",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return jsonify({'error': f'Failed to close position: {result.comment}'}), 500
            
        return jsonify({'status': 'success', 'message': 'Position closed successfully'})
        
    except Exception as e:
        print(f"Error closing position: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def monitor_live_trading():
    global live_trading_active, last_signal
    
    try:
        # Define base pairs
        base_pairs = ['GBPUSD', 'EURUSD', 'NZDUSD', 'AUDUSD']
        
        # Get available symbols
        symbols = mt5.symbols_get()
        available_symbols = [symbol.name for symbol in symbols] if symbols else []
        
        # Find the correct symbol names
        pairs = []
        for pair in base_pairs:
            if f"{pair}m" in available_symbols:
                pairs.append(f"{pair}m")
            elif pair in available_symbols:
                pairs.append(pair)
            else:
                print(f"Warning: Symbol {pair} not found in MT5")
        
        if not pairs:
            print("Error: No valid currency pairs found. Stopping live trading.")
            live_trading_active = False
            return
            
        last_signal = 'None'
        signal = 'None'
        
        print(f"Starting live trading monitor with pairs: {pairs}")
        
        while live_trading_active:
            try:
                # Fetch recent data for all pairs
                pair_data = {}
                for pair in pairs:
                    # Get enough bars to calculate indicators
                    bars_needed = max(live_trading_params['emaPeriod'] * 2, 300)
                    rates = mt5.copy_rates_from_pos(pair, live_trading_params['mt5_timeframe'], 0, bars_needed)
                    if rates is None:
                        print(f"Failed to get data for {pair}")
                        continue
                        
                    if len(rates) == 0:
                        print(f"No data available for {pair}")
                        continue
                        
                    pair_data[pair] = pd.DataFrame(rates)
                    # Ensure time is converted to datetime objects
                    pair_data[pair]['time'] = pd.to_datetime(pair_data[pair]['time'], unit='s')
                
                # Calculate basket price only if we have data for all pairs
                if len(pair_data) == len(pairs) and all(len(pair_data[pair]) > 0 for pair in pairs):
                    # Create basket dataframe
                    basket_df = pd.DataFrame()
                    basket_df['time'] = pair_data[pairs[0]]['time']
                    basket_df['basket_price'] = sum(pair_data[pair]['close'] for pair in pairs) / len(pairs)
                    
                    # Calculate EMA
                    basket_df['ema'] = basket_df['basket_price'].ewm(span=live_trading_params['emaPeriod'], adjust=False).mean()
                    
                    # Check for crossover signal on the most recent 2 bars
                    if len(basket_df) >= 2:
                        current_bar = basket_df.iloc[-1]
                        prev_bar = basket_df.iloc[-2]
                        
                        # Make sure these are float comparisons
                        current_price = float(current_bar['basket_price'])
                        current_ema = float(current_bar['ema'])
                        prev_price = float(prev_bar['basket_price'])
                        prev_ema = float(prev_bar['ema'])
                        
                        # Check for crossover using explicit float comparisons
                        cross_above_ema = (prev_price <= prev_ema) and (current_price > current_ema)
                        cross_below_ema = (prev_price >= prev_ema) and (current_price < current_ema)
                        
                        # Generate signals
                        if cross_above_ema:
                            signal = 'buy'
                            last_signal = 'Buy Signal - Cross Above EMA'
                            print(f"BUY SIGNAL: Basket price {current_price} crossed above EMA {current_ema}")
                            
                            if live_trading_params['tradingMode'] == 'auto':
                                # First close any sell positions
                                for pair in pairs:
                                    positions = mt5.positions_get(symbol=pair)
                                    if positions:
                                        for position in positions:
                                            if position.type == 1:  # Sell position
                                                close_request = {
                                                    "action": mt5.TRADE_ACTION_DEAL,
                                                    "symbol": position.symbol,
                                                    "volume": position.volume,
                                                    "type": mt5.ORDER_TYPE_BUY,
                                                    "position": position.ticket,
                                                    "price": mt5.symbol_info_tick(position.symbol).ask,
                                                    "deviation": 20,
                                                    "magic": 123456,
                                                    "comment": "Close sell position",
                                                    "type_time": mt5.ORDER_TIME_GTC,
                                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                                }
                                                result = mt5.order_send(close_request)
                                                print(f"Closed sell position: {result.retcode}")
                                
                                # Then open buy positions
                                open_basket_positions('buy', pairs, live_trading_params['riskPerTrade'])
                                
                        elif cross_below_ema:
                            signal = 'sell'
                            last_signal = 'Sell Signal - Cross Below EMA'
                            print(f"SELL SIGNAL: Basket price {current_price} crossed below EMA {current_ema}")
                            
                            if live_trading_params['tradingMode'] == 'auto':
                                # First close any buy positions
                                for pair in pairs:
                                    positions = mt5.positions_get(symbol=pair)
                                    if positions:
                                        for position in positions:
                                            if position.type == 0:  # Buy position
                                                close_request = {
                                                    "action": mt5.TRADE_ACTION_DEAL,
                                                    "symbol": position.symbol,
                                                    "volume": position.volume,
                                                    "type": mt5.ORDER_TYPE_SELL,
                                                    "position": position.ticket,
                                                    "price": mt5.symbol_info_tick(position.symbol).bid,
                                                    "deviation": 20,
                                                    "magic": 123456,
                                                    "comment": "Close buy position",
                                                    "type_time": mt5.ORDER_TIME_GTC,
                                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                                }
                                                result = mt5.order_send(close_request)
                                                print(f"Closed buy position: {result.retcode}")
                                
                                # Then open sell positions
                                open_basket_positions('sell', pairs, live_trading_params['riskPerTrade'])
                else:
                    print("Couldn't calculate basket price: missing data for some pairs")
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                print(f"Error in live trading monitor: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # Wait before retrying
    
    except Exception as e:
        print(f"Fatal error in monitor_live_trading: {str(e)}")
        import traceback
        traceback.print_exc()
        live_trading_active = False

def open_basket_positions(direction, pairs, risk_percent):
    try:
        # Calculate position sizes based on risk
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return
        
        # Get account balance and calculate risk amount
        balance = account_info.balance
        risk_amount = balance * risk_percent / 100
        
        # Risk amount per position
        position_risk = risk_amount / len(pairs)
        
        print(f"Opening {direction} positions for {pairs} with {risk_percent}% risk")
        
        for pair in pairs:
            # Get symbol info
            symbol_info = mt5.symbol_info(pair)
            if symbol_info is None:
                print(f"Failed to get symbol info for {pair}")
                continue
                
            if not symbol_info.visible:
                print(f"Symbol {pair} is not visible, trying to select it")
                if not mt5.symbol_select(pair, True):
                    print(f"Failed to select symbol {pair}")
                    continue
            
            # Get tick value and calculate position size
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            
            if tick_size == 0 or tick_value == 0:
                print(f"Invalid tick size or value for {pair}")
                continue
                
            # Get current price
            current_price = None
            if direction == 'buy':
                current_price = symbol_info.ask
            else:
                current_price = symbol_info.bid
                
            if current_price is None or current_price == 0:
                print(f"Failed to get price for {pair}")
                continue
            
            # Calculate lot size based on risk (simplified - should be improved for production)
            point_value = tick_value / tick_size  # Value of 1 point
            one_lot_risk = 1000 * point_value     # Risk per lot (1000 points risk)
            lots = position_risk / one_lot_risk    # Risk amount / risk per lot
            
            # Ensure lot size is within allowed range
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            step_lot = symbol_info.volume_step
            
            lots = max(min_lot, min(max_lot, lots))
            lots = round(lots / step_lot) * step_lot
            
            print(f"Opening {direction} position for {pair} with lot size {lots}")
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pair,
                "volume": lots,
                "type": mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Basket index strategy",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Failed to open position for {pair}: {result.comment}")
            else:
                print(f"Successfully opened {direction} position for {pair}")
                
    except Exception as e:
        print(f"Error opening basket positions: {str(e)}")
        traceback.print_exc()

def calculate_signal_strength(basket_value, basket_atr):
    try:
        # Calculate signal strength based on distance from ATR bands
        atr_band_up = basket_value + live_trading_params['atrMultiplier'] * basket_atr
        atr_band_down = basket_value - live_trading_params['atrMultiplier'] * basket_atr
        
        # Calculate distance from current price to nearest band
        distance_up = abs(basket_value - atr_band_up)
        distance_down = abs(basket_value - atr_band_down)
        min_distance = min(distance_up, distance_down)
        
        # Convert distance to strength percentage
        band_width = atr_band_up - atr_band_down
        strength = (1 - (min_distance / band_width)) * 100
        
        return max(0, min(100, strength))  # Ensure between 0 and 100
        
    except Exception as e:
        print(f"Error calculating signal strength: {str(e)}")
        return 0

def get_market_data(symbol):
    """Fetch market data for a given symbol"""
    try:
        # Try yfinance first
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return {
                'price': data['Close'].iloc[-1],
                'change': ((data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100,
                'volume': data['Volume'].iloc[-1]
            }
        
        # Fallback to Alpha Vantage
        data, _ = alpha_vantage.get_quote_endpoint(symbol)
        return {
            'price': float(data['05. price']),
            'change': float(data['10. change percent'].strip('%')),
            'volume': float(data['06. volume'])
        }
    except Exception as e:
        print(f"Error fetching market data for {symbol}: {e}")
        return None

def get_latest_news(query):
    """Fetch latest news articles"""
    try:
        news = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            page_size=5
        )
        return news['articles']
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_sentiment(text):
    """Analyze sentiment of text using GPT-4"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Analyze the sentiment of the following text and provide a brief summary."},
                {"role": "user", "content": text}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return None

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with market data integration"""
    try:
        print("Received chat request")
        
        # Validate request data
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        if not data or 'message' not in data:
            print("Error: No message in request")
            return jsonify({'error': 'No message provided'}), 400
            
        user_message = data['message']
        print(f"Processing message: {user_message}")
        
        # Check if OpenAI API key is set
        if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-openai-api-key':
            print("Error: OpenAI API key not configured")
            return jsonify({'error': 'OpenAI API key not configured'}), 500
        
        # Prepare context with market data
        context = {
            'market_data': {},
            'news': [],
            'analysis': None
        }
        
        # Extract potential symbols from message
        symbols = extract_symbols(user_message)
        print(f"Extracted symbols: {symbols}")
        
        for symbol in symbols:
            print(f"Fetching market data for {symbol}")
            market_data = get_market_data(symbol)
            if market_data:
                context['market_data'][symbol] = market_data
        
        # Get relevant news
        print("Fetching news")
        news = get_latest_news(user_message)
        if news:
            context['news'] = news[:3]  # Include top 3 news articles
            
            # Analyze sentiment of news
            print("Analyzing sentiment")
            news_text = ' '.join([article['title'] + ' ' + article['description'] for article in news])
            sentiment = analyze_sentiment(news_text)
            context['analysis'] = sentiment
        
        # Prepare messages for GPT-4
        messages = [
            {
                "role": "system",
                "content": """You are an advanced trading assistant with expertise in stocks, crypto, and forex markets. 
                Provide accurate, professional analysis based on the latest market data and news. 
                Format responses clearly with markdown and include relevant market data when available."""
            },
            {
                "role": "user",
                "content": f"""User Query: {user_message}
                
                Market Data: {json.dumps(context['market_data'], indent=2)}
                
                Recent News: {json.dumps([n['title'] for n in context['news']], indent=2)}
                
                Sentiment Analysis: {context['analysis']}
                
                Please provide a comprehensive response incorporating this information."""
            }
        ]
        
        print("Sending request to OpenAI")
        # Get response from GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500
        )
        
        print("Received response from OpenAI")
        return jsonify({
            'response': response.choices[0].message.content,
            'context': context
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def extract_symbols(text):
    """Extract potential trading symbols from text"""
    # Common stock market symbols
    stock_pattern = r'\b[A-Z]{1,5}\b'
    # Crypto patterns (BTC/USD, ETH-USD, etc.)
    crypto_pattern = r'\b[A-Z]{3,4}[-/][A-Z]{3,4}\b'
    # Forex patterns (EUR/USD, GBP/JPY, etc.)
    forex_pattern = r'\b[A-Z]{3}/[A-Z]{3}\b'
    
    symbols = []
    for pattern in [stock_pattern, crypto_pattern, forex_pattern]:
        symbols.extend(re.findall(pattern, text.upper()))
    
    return list(set(symbols))  # Remove duplicates

@app.route('/control_ea', methods=['POST'])
def control_ea():
    global ea_status, live_trading_active
    try:
        # Get JSON data from request
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
            
        action = data.get('action')
        if not action:
            return jsonify({'success': False, 'error': 'No action specified'})
        
        print(f"EA control action received: {action}")
        
        # Initialize MT5 connection
        if not mt5.initialize():
            return jsonify({'success': False, 'error': 'Failed to connect to MetaTrader5'})
        
        # Get terminal info
        terminal = mt5.terminal_info()._asdict()
        if not terminal['trade_allowed']:
            return jsonify({'success': False, 'error': 'Please enable AutoTrading in MT5 first'})
        
        # Get all positions to identify active EAs
        positions = mt5.positions_get()
        active_eas = set()
        active_pairs = set()
        
        if positions:
            for pos in positions:
                if pos.magic == 123456:  # Currently implemented EA magic number
                    active_eas.add(pos.magic)
                    active_pairs.add(pos.symbol)
                    print(f"Found EA position on {pos.symbol}")
        
        if action == 'start':
            try:
                # Enable automated trading
                ea_status = True
                live_trading_active = True
                
                # Log current state
                print("Current MT5 State:")
                print(f"Active EAs: {active_eas}")
                print(f"Trading pairs: {active_pairs}")
                print("EA Status: Enabled")
                print("Live Trading: Enabled")
                
                return jsonify({
                    'success': True, 
                    'message': 'EA trading enabled',
                    'active_eas': list(active_eas),
                    'trading_pairs': list(active_pairs)
                })
                
            except Exception as e:
                print(f"Error starting EA: {str(e)}")
                return jsonify({'success': False, 'error': f'Failed to start EA: {str(e)}'})
                
        elif action == 'stop':
            try:
                # First disable EA trading flags
                ea_status = False
                live_trading_active = False
                
                print("EA Status: Disabled")
                print("Live Trading: Disabled")
                
                # Close all positions for EA
                closed_positions = []
                failed_closes = []
                
                if positions:
                    for position in positions:
                        if position.magic == 123456:  # Only close EA positions
                            try:
                                close_request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": position.symbol,
                                    "volume": position.volume,
                                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                                    "position": position.ticket,
                                    "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
                                    "deviation": 20,
                                    "magic": 123456,
                                    "comment": "EA Stop - Close Position",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                                
                                # Send close order
                                result = mt5.order_send(close_request)
                                print(f"Closing position {position.ticket} result: {result.retcode} for {position.symbol}")
                                
                                if result.retcode == mt5.TRADE_RETCODE_DONE:
                                    closed_positions.append(position.ticket)
                                else:
                                    failed_closes.append({
                                        'ticket': position.ticket,
                                        'symbol': position.symbol,
                                        'error': result.comment
                                    })
                                    
                            except Exception as close_error:
                                print(f"Error closing position {position.ticket}: {str(close_error)}")
                                failed_closes.append({
                                    'ticket': position.ticket,
                                    'symbol': position.symbol,
                                    'error': str(close_error)
                                })
                
                # Verify all positions are closed
                remaining_positions = mt5.positions_get()
                remaining_ea_positions = [pos for pos in remaining_positions if pos.magic == 123456] if remaining_positions else []
                
                response = {
                    'success': True,
                    'message': 'EA trading disabled',
                    'closed_positions': closed_positions,
                    'failed_closes': failed_closes,
                    'remaining_positions': len(remaining_ea_positions),
                    'closed_eas': list(active_eas),
                    'affected_pairs': list(active_pairs)
                }
                
                print(f"Stop EA Response: {response}")
                return jsonify(response)
                
            except Exception as e:
                print(f"Error stopping EA: {str(e)}")
                return jsonify({'success': False, 'error': f'Failed to stop EA: {str(e)}'})
        else:
            return jsonify({'success': False, 'error': 'Invalid action'})
            
    except Exception as e:
        print(f"Error in control_ea: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        mt5.shutdown()

@app.route('/ea_status')
def get_ea_status():
    global ea_status, live_trading_active
    try:
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return jsonify({'is_running': False})
            
        # Get terminal info
        terminal_info = mt5.terminal_info()._asdict()
        print(f"MT5 Terminal Info: {terminal_info}")
        
        if not terminal_info['trade_allowed']:
            print("AutoTrading is disabled in MT5")
            return jsonify({'is_running': False, 'message': 'AutoTrading is disabled in MT5'})
        
        # Get all positions
        positions = mt5.positions_get()
        print(f"\nActive Positions: {len(positions) if positions else 0}")
        
        active_eas = {}
        if positions:
            for pos in positions:
                print(f"\nPosition Details:")
                print(f"Symbol: {pos.symbol}")
                print(f"Magic Number: {pos.magic}")
                print(f"Type: {'Buy' if pos.type == 0 else 'Sell'}")
                print(f"Volume: {pos.volume}")
                print(f"Profit: {pos.profit}")
                print(f"Comment: {pos.comment}")
                
                if pos.magic == 123456:  # Check for our EA's magic number
                    if pos.magic not in active_eas:
                        active_eas[pos.magic] = {
                            'pairs': set(),
                            'positions': 0,
                            'total_volume': 0,
                            'total_profit': 0,
                            'comments': set()
                        }
                    ea_info = active_eas[pos.magic]
                    ea_info['pairs'].add(pos.symbol)
                    ea_info['positions'] += 1
                    ea_info['total_volume'] += pos.volume
                    ea_info['total_profit'] += pos.profit
                    if pos.comment:
                        ea_info['comments'].add(pos.comment)
        
        # Convert sets to lists for JSON serialization
        for magic in active_eas:
            active_eas[magic]['pairs'] = list(active_eas[magic]['pairs'])
            active_eas[magic]['comments'] = list(active_eas[magic]['comments'])
        
        is_running = ea_status and live_trading_active
        
        response_data = {
            'is_running': is_running,
            'ea_status': ea_status,
            'live_trading_active': live_trading_active,
            'autotrading_enabled': terminal_info['trade_allowed'],
            'active_eas': active_eas,
            'message': f"Found {len(active_eas)} active EAs" if active_eas else 'No active EAs detected',
            'terminal_info': {
                'trade_allowed': terminal_info['trade_allowed'],
                'connected': terminal_info['connected'],
                'ea_enabled': terminal_info['trade_allowed']
            }
        }
        
        print("\nResponse Data:", json.dumps(response_data, indent=2))
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in ea_status: {str(e)}")
        traceback.print_exc()
        return jsonify({'is_running': False, 'error': str(e)})
    finally:
        mt5.shutdown()

@app.route('/api/mt5_info')
def get_mt5_info():
    try:
        print("Attempting to initialize MT5...")
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return jsonify({'error': 'Failed to initialize MT5'})
        print("MT5 initialized successfully")
            
        # Get terminal info
        print("Getting terminal info...")
        terminal_info = mt5.terminal_info()._asdict()
        print(f"Terminal info: {terminal_info}")
        
        # Get account info
        print("Getting account info...")
        account_info = mt5.account_info()._asdict()
        print(f"Account info: {account_info}")
        
        # Get all positions
        print("Getting positions...")
        positions = mt5.positions_get()
        position_info = []
        active_eas = set()
        
        if positions:
            print(f"Found {len(positions)} positions")
            for pos in positions:
                pos_data = {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'magic': pos.magic,
                    'type': 'Buy' if pos.type == 0 else 'Sell',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'comment': pos.comment,
                    'time': datetime.fromtimestamp(pos.time).strftime('%Y-%m-%d %H:%M:%S')
                }
                print(f"Position data: {pos_data}")
                position_info.append(pos_data)
                
                if pos.magic > 0:
                    active_eas.add(pos.magic)
        else:
            print("No positions found")
        
        # Get all symbols
        print("Getting symbols...")
        symbols = mt5.symbols_get()
        symbol_info = []
        if symbols:
            print(f"Found {len(symbols)} symbols")
            for symbol in symbols:
                if symbol.trade_mode != 0:
                    symbol_info.append({
                        'name': symbol.name,
                        'description': symbol.description,
                        'path': symbol.path,
                        'trade_mode': symbol.trade_mode
                    })
        else:
            print("No symbols found")
        
        response_data = {
            'terminal_info': {
                'connected': terminal_info['connected'],
                'trade_allowed': terminal_info['trade_allowed'],
                'ea_enabled': terminal_info['trade_allowed'],
                'dlls_allowed': terminal_info['dlls_allowed']
            },
            'account_info': {
                'login': account_info['login'],
                'server': account_info['server'],
                'balance': account_info['balance'],
                'equity': account_info['equity'],
                'margin': account_info['margin'],
                'margin_free': account_info['margin_free']
            },
            'positions': position_info,
            'active_eas': list(active_eas),
            'symbols': symbol_info
        }
        print(f"Sending response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error getting MT5 info: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)})
    finally:
        print("Shutting down MT5...")
        mt5.shutdown()

if __name__ == '__main__':
    try:
        print("\nChecking MT5 status on startup...")
        if not mt5.initialize():
            print("Failed to initialize MT5")
        else:
            print("MT5 initialized successfully")
            
            terminal_info = mt5.terminal_info()._asdict()
            print(f"\nTerminal Info:")
            print(f"Connected: {terminal_info['connected']}")
            print(f"Trade Allowed: {terminal_info['trade_allowed']}")
            print(f"EA Enabled: {terminal_info['trade_allowed']}")
            
            account_info = mt5.account_info()._asdict()
            print(f"\nAccount Info:")
            print(f"Login: {account_info['login']}")
            print(f"Server: {account_info['server']}")
            print(f"Balance: {account_info['balance']}")
            print(f"Equity: {account_info['equity']}")
            
            positions = mt5.positions_get()
            if positions:
                print(f"\nActive Positions ({len(positions)}):")
                for pos in positions:
                    print(f"Symbol: {pos.symbol}, Type: {'Buy' if pos.type == 0 else 'Sell'}, Magic: {pos.magic}")
            else:
                print("\nNo active positions")
                
            mt5.shutdown()
            
    except Exception as e:
        print(f"Error checking MT5 status: {str(e)}")
        traceback.print_exc()
    
    app.run(**SERVER_CONFIG) 
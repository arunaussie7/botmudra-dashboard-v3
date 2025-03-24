# Comprehensive Documentation: Forex Trading Platform

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Backend Architecture (app.py)](#3-backend-architecture-apppy)
4. [Frontend Components](#4-frontend-components)
   - [Dashboard Interface](#41-dashboard-interface)
   - [Backtester Interface](#42-backtester-interface)
   - [Currency Strength Analysis](#43-currency-strength-analysis)
   - [Correlation Analysis](#44-correlation-analysis)
   - [Pair Movements](#45-pair-movements)
   - [Market Movements](#46-market-movements)
5. [Trading Features](#5-trading-features)
   - [Basket Trading Strategy](#51-basket-trading-strategy)
   - [Backtesting Engine](#52-backtesting-engine)
   - [Live Trading](#53-live-trading)
6. [Setup and Installation](#6-setup-and-installation)
7. [Configuration](#7-configuration)
8. [Troubleshooting Guide](#8-troubleshooting-guide)

## 1. Project Overview

This is a comprehensive forex trading platform that connects to MetaTrader 5 (MT5) to provide real-time trading capabilities, advanced market analysis, and trade management. The platform features:

- Basket trading strategy implementation with EMA crossover signals
- Advanced backtesting capabilities with detailed performance analytics
- Live trading automation with risk management controls
- Currency strength analysis tools
- Correlation matrix visualization
- Pair movements analysis
- Real-time market data visualization

## 2. Project Structure

```
Trading-website/
├── app.py                      # Main Flask application (backend)
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── templates/                  # HTML templates (frontend)
│   ├── index.html              # Landing page
│   ├── dashboard.html          # Main dashboard
│   ├── backtester.html         # Backtesting interface
│   ├── strength.html           # Currency strength analyzer
│   ├── correlation.html        # Correlation matrix
│   ├── pair_movements.html     # Pair movement analyzer
│   └── market_movements.html   # Market movements analyzer
├── static/                     # Static assets
│   ├── css/                    # Stylesheets
│   ├── js/                     # JavaScript files
│   └── images/                 # Image assets
└── uploaded_eas/               # Uploaded Expert Advisors
```

## 3. Backend Architecture (app.py)

The backend is built with Flask and integrates with MetaTrader 5 (MT5) for trading operations. The main components include:

### Core Initialization
- **Lines 1-50**: Import statements, environment setup, MT5 configuration
- **MT5 Connection**: Establishes connection to the trading platform
- **Global Variables**: Manages live trading state, parameters, and positions

### API Endpoints

#### Main Route Handlers
- **/** - Landing page
- **/dashboard** - Main dashboard interface
- **/backtester** - Backtesting interface
- **/correlation** - Correlation analysis
- **/strength** - Currency strength analysis
- **/pair_movements** - Pair movements
- **/market_movements** - Market movements analysis

#### Data API Endpoints
- **/api/account_info** - Returns account balance, equity, margin
- **/api/symbols** - Returns list of available currency pairs
- **/api/correlation_data** - Returns correlation matrix data
- **/api/strength_data** - Returns currency strength metrics
- **/api/basket_backtest** - Processes backtesting requests
- **/api/start_live_trading** - Initiates live trading
- **/api/stop_live_trading** - Stops live trading
- **/api/live_trading_status** - Returns current trading status and positions

### Key Functions

#### Basket Strategy Implementation
- **basket_backtest()** (Lines 1400-1700): Implementation of the basket trading strategy backtester
  - Takes parameters: timeframe, start/end dates, initial balance, lot size, emaPeriod
  - Uses MT5 to fetch historical data for currency pairs
  - Calculates basket index and EMA
  - Generates trade signals based on EMA crossovers
  - Tracks performance and calculates statistics

#### Live Trading Functions
- **monitor_live_trading()** (Lines 1850-2000): Manages the live trading thread
  - Periodically checks for EMA crossovers on the basket index
  - Generates buy/sell signals
  - Manages position opening and closing

- **open_basket_positions()** (Lines 2000-2100): Handles opening positions
  - Calculates position sizes based on risk parameters
  - Sends orders to MT5
  - Handles order validation and execution

## 4. Frontend Components

### 4.1 Dashboard Interface
**File Location**: `templates/dashboard.html`

The dashboard is the main interface for the trading platform. It provides navigation to all analysis tools and displays account information.

**Key Features**:
- Navigation sidebar with links to all tools
- Account information display
- Latest price updates and market news

**Code Structure**:
- Navigation menu: Lines 50-100
- Account information display: Lines 100-150
- Market news widget: Lines 150-200

### 4.2 Backtester Interface
**File Location**: `templates/backtester.html`

The backtester allows users to test the basket trading strategy using historical data. It provides detailed performance metrics and visualizations.

**Key Features**:
- Input forms for strategy parameters:
  - Timeframe selection (M1 to MN1)
  - Date range selection
  - Initial balance setting
  - EMA period configuration
  - Lot size setting
- Detailed results display:
  - Net profit, win rate, profit factor, max drawdown statistics
  - Equity curve chart
  - Basket price vs. EMA chart
  - Detailed basket transactions table with:
    - Direction (BUY/SELL)
    - Entry/Exit dates and times
    - Trade duration
    - Total profit
    - Individual currency pair profits
  - Currency pair performance analysis:
    - Net profit per pair
    - Win rate per pair
    - Contribution to overall performance

**Code Structure**:
- Parameter input forms: Lines 300-500
- Results statistics display: Lines 500-650
- Charts: Lines 650-800
- Detailed transactions table: Lines 800-900
- Pair performance table: Lines 900-1000

**JavaScript Functions**:
- `runBacktest()`: Sends parameters to backend API, Lines 1000-1050
- `displayResults()`: Processes and displays results, Lines 1050-1300
- Date handling: Lines 700-730

### 4.3 Currency Strength Analysis
**File Location**: `templates/strength.html`

Displays the relative strength of major currencies across different timeframes.

**Key Features**:
- Strength indicators for 8 major currencies
- Multi-timeframe analysis
- Historical strength charts
- Currency comparisons

**Code Structure**:
- Currency selector: Lines 50-100
- Timeframe selector: Lines 100-150
- Strength visualization: Lines 150-300
- Historical data charts: Lines 300-500

### 4.4 Correlation Analysis
**File Location**: `templates/correlation.html`

Visualizes the correlation between different currency pairs.

**Key Features**:
- Correlation matrix heatmap
- Correlation coefficient display
- Timeframe selection
- Pair comparison tools

**Code Structure**:
- Correlation matrix visualization: Lines 100-300
- Timeframe selector: Lines 50-100
- Pair selector: Lines 300-350
- Data processing: Lines 350-500

### 4.5 Pair Movements
**File Location**: `templates/pair_movements.html`

Analyzes price movements across different currency pairs.

**Key Features**:
- Price movement comparison
- Volatility analysis
- Trend direction indicators
- Multi-timeframe analysis

**Code Structure**:
- Pair selector: Lines 50-80
- Timeframe selector: Lines 80-110
- Movement visualization: Lines 110-200

### 4.6 Market Movements
**File Location**: `templates/market_movements.html`

Provides an overview of market-wide movements and trends.

**Key Features**:
- Market overview dashboard
- Top movers identification
- Trend analysis
- Multi-timeframe scanning

**Code Structure**:
- Market overview: Lines 50-150
- Top movers table: Lines 150-200
- Trend analysis visualization: Lines 200-300

## 5. Trading Features

### 5.1 Basket Trading Strategy
The platform implements a basket trading strategy that tracks the average price of multiple currency pairs (GBPUSD, EURUSD, NZDUSD, AUDUSD) and generates signals based on EMA crossovers.

**Strategy Logic**:
1. Calculate basket index (average price of selected pairs)
2. Calculate EMA of the basket index
3. Generate BUY signal when basket price crosses above EMA
4. Generate SELL signal when basket price crosses below EMA
5. Close opposite positions when a new signal is generated

**Implementation**:
- Basket calculation: `app.py`, Lines 1400-1450
- Signal generation: `app.py`, Lines 1450-1500
- Position management: `app.py`, Lines 1500-1600

### 5.2 Backtesting Engine
The backtesting engine allows testing the strategy on historical data with detailed performance metrics.

**Features**:
- Historical data retrieval from MT5
- Signal generation based on strategy rules
- Performance tracking and statistics calculation
- Detailed trade analysis including individual pair performance

**Implementation**:
- Data retrieval: `app.py`, Lines 1450-1500
- Strategy execution: `app.py`, Lines 1500-1650
- Performance calculation: `app.py`, Lines 1650-1700

### 5.3 Live Trading
The live trading module connects to MT5 to execute trades based on the strategy signals.

**Features**:
- Real-time data monitoring
- Automatic signal generation
- Position sizing based on risk parameters
- Position management (opening/closing)

**Implementation**:
- Live monitoring: `app.py`, Lines 1850-2000
- Position opening: `app.py`, Lines 2000-2100
- Position closing: `app.py`, Lines 2100-2150

## 6. Setup and Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure MT5**:
   - Ensure MT5 is installed and running
   - Update MT5 credentials in `app.py` (Lines 25-30)

3. **Set Up Environment Variables**:
   - Create a `.env` file with:
     ```
     ALPHA_VANTAGE_API_KEY=your_key_here
     NEWS_API_KEY=your_key_here
     PORT=5000
     ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Platform**:
   Open a web browser and navigate to `http://localhost:5000`

## 7. Configuration

### MT5 Connection
Configure MT5 connection parameters in `app.py` (Lines 25-30):
```python
MT5_CONFIG = {
    "server": "Your_MT5_Server",
    "login": your_login_number,
    "password": "your_password"
}
```

### API Keys
Set up API keys in `.env` file:
```
ALPHA_VANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

### Strategy Parameters
Default strategy parameters can be modified in the backtester interface:
- EMA Period: Default 200
- Lot Size: Default 0.01
- Initial Balance: Default 10000

## 8. Troubleshooting Guide

### Common Issues

#### Connection to MT5 Fails
- Ensure MT5 is running
- Verify login credentials
- Check server address

#### Backtester Returns Error
- Verify date range is valid
- Check if selected symbols are available
- Ensure input parameters are within acceptable ranges

#### Live Trading Issues
- Check account balance and margin requirements
- Verify internet connection
- Ensure MT5 is properly connected

#### Input Field Problems
- If you can't type in numeric fields, try refreshing the page
- Make sure you're entering valid values within the specified range

#### Data Retrieval Errors
- Check specific error messages in console
- Verify API keys are valid
- Ensure you have access to the requested historical data

#### Symbol Not Found Errors
- Error: "Failed to get data for [Symbol]"
  - Solution: Check if the symbol exists in your MT5 terminal
  - Try with or without 'm' suffix (e.g., EURUSDm vs EURUSD)
  - Verify that you have historical data for the requested time period

#### Numeric Input Fields Not Working
- Error: Can't type in EMA period, lot size, or initial balance fields
  - Solution: The input restriction has been fixed by removing restrictive event listeners
  - Refresh the page after the fix is applied
  - If problem persists, try using a different browser

## 9. Platform Functionality Details

### 9.1 Basket Index Calculation
The basket index is calculated as the average price of selected currency pairs:
```
Basket Index = (GBPUSD + EURUSD + NZDUSD + AUDUSD) / 4
```

The implementation dynamically adjusts for available symbols, checking for both regular and 'm' suffix versions of each pair.

### 9.2 Signal Generation Logic
```python
# Generate BUY signal
if prev_bar['basket_price'] <= prev_bar['ema'] and current_bar['basket_price'] > current_bar['ema']:
    # BUY signal logic
    
# Generate SELL signal
if prev_bar['basket_price'] >= prev_bar['ema'] and current_bar['basket_price'] < current_bar['ema']:
    # SELL signal logic
```

### 9.3 Performance Metrics Calculation
The backtest engine calculates various performance metrics:

- **Win Rate**: Percentage of profitable trades
  ```
  Win Rate = (Profitable Trades / Total Trades) * 100
  ```

- **Profit Factor**: Ratio of gross profit to gross loss
  ```
  Profit Factor = Gross Profit / Gross Loss
  ```

- **Max Drawdown**: Maximum peak-to-trough decline in equity
  ```
  Max Drawdown = ((Max Equity - Min Equity) / Max Equity) * 100
  ```

### 9.4 Detailed Trade Analysis
For each trade, the system records:
- Entry and exit dates
- Position direction (BUY/SELL)
- Trade duration
- Total profit/loss
- Individual profit/loss for each currency pair
- Exit reason (opposite signal, end of test)

## 10. Future Enhancements

### 10.1 Planned Features
- **Advanced Risk Management**: Additional risk control parameters
- **Multi-Strategy Backtesting**: Support for multiple strategy comparison
- **Strategy Optimization**: Automated parameter optimization
- **Custom Indicators**: User-defined technical indicators
- **Performance Reporting**: Enhanced reporting capabilities with export options

### 10.2 Extensibility
The platform is designed to be extensible:
- New strategies can be added by implementing new functions in app.py
- Additional analysis tools can be created by adding new templates and routes
- Custom indicators can be implemented by extending the technical analysis libraries

## 11. Contact and Support

For issues or questions, please contact the development team.

---

*This documentation was last updated on March 19, 2023.* 
# Botmudra Dashboard v3

A comprehensive MT5 trading dashboard with advanced features for automated trading, market analysis, and portfolio management.

## Features

- **EA Control Panel**: Start/Stop EA trading with real-time status monitoring
- **Basket Trading Strategy**: Implements a sophisticated basket trading approach across multiple currency pairs
- **Market Analysis Tools**:
  - Currency Strength Meter
  - Correlation Analysis
  - Pair Movements Analysis
  - Market Movements Dashboard
- **Live Trading Features**:
  - Real-time position monitoring
  - Automated trade execution
  - Risk management controls
- **Backtesting Module**: Test strategies with historical data
- **Account Management**: Monitor account metrics, positions, and trading history

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/botmudra-dashboard-v3.git
cd botmudra-dashboard-v3
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure your MT5 credentials in `config.py`

## Configuration

1. Create a `config.py` file with your MT5 credentials:
```python
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 8080,
    'debug': True
}

MT5_CONFIG = {
    "server": "Your-MT5-Server",
    "login": Your-Login-Number,
    "password": "Your-Password"
}
```

2. Set up environment variables for API keys (optional):
- ALPHA_VANTAGE_API_KEY
- NEWS_API_KEY
- OPENAI_API_KEY

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the dashboard at `http://localhost:8080`

3. Use the various features:
- Monitor EA status
- Control trading operations
- Analyze market data
- Run backtests
- View account statistics

## Requirements

- Python 3.8+
- MetaTrader 5
- Flask
- Pandas
- NumPy
- Other dependencies listed in requirements.txt

## License

MIT License

## Version History

- v3.0.0: Initial release with basket trading strategy and comprehensive dashboard
- Future updates will be listed here

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 
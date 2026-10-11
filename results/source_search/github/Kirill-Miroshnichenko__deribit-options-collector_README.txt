# 📊 Deribit Options Data Collector

Real-time cryptocurrency options data collector for Deribit exchange for quantitative analysis and algorithmic trading.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

- **Real-time Data Collection**: Capture live options data at configurable intervals
- **Complete Market Data**: 
  - Option prices (mark, bid, ask, last, mid)
  - Greeks (delta, gamma, vega, theta, rho)
  - Implied Volatility (mark IV, bid IV, ask IV)
  - Open Interest & 24h Volume
  - Underlying price tracking
- **Efficient Storage**: Compressed Parquet format for optimal performance
- **Multiple Currencies**: Support for BTC, ETH and other cryptocurrencies
- **Data Accumulation**: Automatic appending to current day files

## 📈 Use Cases

- **Algorithmic Trading**: Build and backtest options strategies (wheel, iron condor, straddles)
- **Volatility Analysis**: Track IV/HV relationships and build volatility surfaces
- **Market Research**: Analyze option flow, Greeks evolution and market microstructure
- **Risk Management**: Monitor portfolio Greeks and volatility exposure

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deribit-options-collector.git
cd deribit-options-collector

# Install dependencies
pip install -r requirements.txt
```

### Single Data Collection

```python
from collector import DeribitDataCollector

# Initialize collector
collector = DeribitDataCollector(currency='ETH', data_dir='deribit_data')

# Collect data
df = collector.collect_options_data()

# Save to Parquet
collector.save_to_parquet(df)
```

### Periodic Collection

```python
from collector import periodic_collection

# Collect every 2 minutes, 30 iterations (1 hour)
periodic_collection(
    currency='ETH',
    interval_minutes=2,
    iterations=30
)
```

### Load Saved Data

```python
# Load all data
all_data = collector.load_data()

# Load data for specific period
from datetime import datetime, timedelta

start = datetime.now() - timedelta(days=7)
end = datetime.now()
weekly_data = collector.load_data(start_date=start, end_date=end)
```

## 📊 Data Structure

| Column | Type | Description |
|---------|-----|----------|
| `timestamp` | datetime | UTC timestamp of collection |
| `instrument_name` | string | Option identifier (e.g., ETH-28JAN26-3000-P) |
| `strike` | float | Strike price |
| `option_type` | string | 'call' or 'put' |
| `mark_price` | float | Mark price in underlying currency |
| `bid_price` | float | Best bid price |
| `ask_price` | float | Best ask price |
| `mid_price` | float | Mid price (bid + ask) / 2 |
| `delta` | float | Delta (∂V/∂S) |
| `gamma` | float | Gamma (∂²V/∂S²) |
| `vega` | float | Vega (∂V/∂σ) |
| `theta` | float | Theta (∂V/∂t) |
| `rho` | float | Rho (∂V/∂r) |
| `mark_iv` | float | Mark implied volatility |
| `bid_iv` | float | Bid implied volatility |
| `ask_iv` | float | Ask implied volatility |
| `open_interest` | float | Open interest |
| `volume_24h` | float | 24-hour trading volume |
| `underlying_price` | float | Spot price of underlying asset |

## 📁 Project Structure

```
deribit-options-collector/
├── README.md                 # Documentation
├── collector.py              # Main data collector
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore rules
├── examples/                # Usage examples
│   ├── basic_usage.py       # Basic example
│   └── analysis.py          # Data analysis
└── deribit_data/            # Data directory (gitignored)
```

## 📊 Analysis Examples

### Filter Liquid Options

```python
# Only options with non-zero OI and spread
df_liquid = df[
    (df['bid_price'].notna()) & 
    (df['ask_price'].notna()) &
    (df['open_interest'] > 0)
]

print(f"Liquid options: {len(df_liquid)}")
```

### Analysis by Option Type

```python
print(f"Calls: {len(df[df['option_type'] == 'call'])}")
print(f"Puts: {len(df[df['option_type'] == 'put'])}")
print(f"Unique expirations: {df['expiration_timestamp'].nunique()}")
print(f"Unique strikes: {df['strike'].nunique()}")
```

### Build Volatility Smile

```python
import matplotlib.pyplot as plt

# Filter data for specific expiration
expiration = df['expiration_timestamp'].min()
exp_data = df[df['expiration_timestamp'] == expiration]

# Separate calls and puts
calls = exp_data[exp_data['option_type'] == 'call']
puts = exp_data[exp_data['option_type'] == 'put']

# Plot
plt.scatter(calls['strike'], calls['mark_iv']*100, label='Calls', alpha=0.6)
plt.scatter(puts['strike'], puts['mark_iv']*100, label='Puts', alpha=0.6)
plt.xlabel('Strike')
plt.ylabel('Implied Volatility (%)')
plt.title('Volatility Smile')
plt.legend()
plt.show()
```

## 🎓 Technical Details

### Data Collection Architecture

```
Deribit API → Collector → Processing → Parquet Storage
     ↓            ↓            ↓            ↓
  REST API    Python       pandas      Compressed
  Public      Class                     Files
  Endpoint
```

### Performance

- **Collection speed**: ~0.5 second delay per 10 instruments
- **Storage efficiency**: ~50MB per day (compressed format)
- **Memory usage**: <200MB during collection
- **API calls**: Configurable frequency with rate limiting

## 📚 Requirements

```
pandas>=2.0.0
pyarrow>=12.0.0
requests>=2.31.0
```

## 🔄 Data Quality

- **Completeness**: All available fields captured
- **Accuracy**: Data directly from exchange API
- **Validation**: Automatic checking for missing fields
- **Deduplication**: Correct appending to existing files

## 📈 Dataset Statistics

**Real dataset** (example usage):
- **5.3M+ records** collected over 20 days
- **1,866 unique options** tracked
- **Zero gaps** - complete tick-by-tick coverage
- **22 data fields** per record including full Greeks

## 🤝 Contributing

Pull requests are welcome! Please feel free to submit a PR.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code
flake8 collector.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Options trading carries significant risk. Always do your own research and consult with financial professionals before trading.

## 📧 Contact

- GitHub: [@Kirill-Miroshnichenko](https://github.com/Kirill-Miroshnichenko)
- Email: mkv1986@gmail.com

## 🌟 Acknowledgments

- Deribit API for providing comprehensive market data
- Quantitative finance community for inspiration and feedback

---

**Star ⭐ this repository if you find it useful!**

## 🔧 Collection Setup

### Basic Configuration

```python
# Change currency
collector = DeribitDataCollector(currency='BTC')  # or 'ETH'

# Change data directory
collector = DeribitDataCollector(data_dir='my_data')
```

### Periodic Collection 24/7

```python
# Collect every 2 minutes indefinitely
periodic_collection(
    currency='ETH',
    interval_minutes=2,
    iterations=999999  # Essentially infinite
)
```

### Run in Background (Linux/Mac)

```bash
# Run in background
nohup python -c "from collector import periodic_collection; periodic_collection('ETH', 2, 999999)" &

# Check process
ps aux | grep python

# Stop
kill <PID>
```

## 📊 Data Size

Approximate estimates:
- **1 snapshot**: ~50KB (100 options)
- **1 hour** (30 collections): ~1.5MB
- **1 day**: ~36MB
- **1 month**: ~1GB (compressed)

Parquet format provides:
- ~80% compression vs CSV
- Fast loading
- Pandas support

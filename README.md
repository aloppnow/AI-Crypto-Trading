# AI Crypto Trading Bot



## Current Portfolio in Coinbase (Example)
<!-- START:PORTFOLIO -->
| Symbol | Quantity | Est. Value ($) |
|---|---:|---:|
| AVAX | 0.04168420 | 0.98 |
| ADA | 9.63337515 | 7.62 |
| SOL | 0.00536486 | 0.95 |
| LINK | 0.10000000 | 2.16 |
| ETH | 0.28013309 | 1200.19 |
| BTC | 0.00023454 | 28.08 |
<!-- END:PORTFOLIO -->

## Recent Trades (last 10) (Example)
<!-- START:TRADE_LOG -->
| Time (UTC) | Action | Symbol | Amount ($) | Limit |
|---|---|---:|---:|---:|
| 2025-08-10T13:13:01.192268 | buy_crypto_price | ADA | 7.77 |  |
| 2025-08-10T12:46:10.704768 | buy_crypto_price | AVAX | 1.00 |  |
| 2025-08-10T12:07:39.008478 | buy_crypto_price | BTC | 28.13 |  |
| 2025-08-09T21:35:05.160177 | buy_crypto_price | ETH | 1.00 |  |
| 2025-08-09T21:04:17.083104 | buy_crypto_price | SOL | 1.00 |  |
| 2025-08-09T20:36:46.379596 | sell_crypto_price | ETH | 1.00 |  |
| 2025-08-09T20:05:03.459931 | sell_crypto_price | ETH | 1.00 |  |
| 2025-08-09T19:33:41.483062 | sell_crypto_price | ETH | 1.00 |  |
| 2025-08-09T19:04:23.492332 | buy_crypto_price | DOGE | 140.00 |  |
| 2025-08-09T18:41:59.158137 | buy_crypto_price | ETH | 1.00 |  |
<!-- END:TRADE_LOG -->

<!-- START:UPDATED -->
_Last updated: 2025-08-11T18:43:52Z_
<!-- END:UPDATED -->


## Description

This repository contains an advanced, automated crypto trading bot that leverages OpenAI's GPT models to make aggressive, short-term trading decisions for a selection of popular cryptocurrencies. The bot is designed for moderate risk tolerance and aims to maximize 1-month returns by analyzing technical indicators, news sentiment, and momentum signals.

## Features

- **Automated Trading**: Executes trades on supported cryptocurrencies (BTC, ETH, SOL, DOGE, ADA, AVAX, LINK, SHIB) using Robinhood and Coinbase.
- **AI-Driven Decisions**: Uses OpenAI GPT models to analyze market data, news sentiment, and historical trends.
- **Risk Management**: Implements stop-loss logic, cool-off rules, and portfolio allocation strategies.
- **News & Sentiment Analysis**: Aggregates and analyzes top news headlines for each crypto asset.
- **Trade Logging**: Records all trades for transparency and strategy improvement.

## Requirements

- Python 3.8+
- Coinbase accounts
- API keys for Coinbase and OpenAI

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AI-Crypto-Trading.git
   cd AI-Crypto-Trading
   ```
2. **Create a `.env` file** in the root directory with the following variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   COINBASE_API_KEY=your_coinbase_api_key
   COINBASE_API_SECRET=your_coinbase_api_secret
   ```
3. **Run the bot:**
   ```bash
   python crypto_agent_pipeline.py
   ```

## How It Works

- Gathers real-time crypto data, account balances, open orders, positions, historical data, and news headlines.
- Sends a comprehensive prompt to the OpenAI GPT model, which returns a single optimal trading command.
- Executes the recommended trade and logs the action.
- Repeats the process every 30 minutes.

## Customization

- You can modify the trading strategy, risk management rules, or supported cryptocurrencies by editing `crypto_agent_pipeline.py`.
- To add new exchanges or data sources, extend the relevant sections of the code.

## Disclaimer

This project is for educational and research purposes only. Trading cryptocurrencies involves significant risk. Use at your own discretion and risk. The authors are not responsible for any financial losses incurred.

## License

MIT License

# swimming_result_analysis

Web scraping and predictive analysis of Japanese swimming competition results using scikit-learn.

## Overview

This project scrapes swimming competition results from [swim.or.jp](https://result.swim.or.jp/), stores them in CSV files, and uses machine learning (Random Forest Regression) to predict performance improvements. It also includes analysis of best-time update rates across teams and events.

## Features

- **Web Scraping**: Collects race results (name, team, event, time, rank) from official tournament pages by tournament ID.
- **Performance Prediction**: Trains a Random Forest model to predict race times based on historical data.
- **Update Rate Analysis**: Calculates and visualizes the rate at which swimmers achieve personal best times, grouped by team affiliation.

## Requirements

- Python 3.8+
- requests
- BeautifulSoup4
- pandas
- scikit-learn
- Matplotlib

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python result_analysis.py
```

The script scrapes results for predefined tournament IDs, performs analysis, and saves an update-rate bar chart (`update_rate.png`).

## Output

- `results_<tournament_id>.csv`: Scraped race results.
- `update_rate.png`: Bar chart of best-time update rates by team.

# Crypto Protocol Valuation Model

A sophisticated DCF-based valuation model for cryptocurrency protocols, featuring Monte Carlo simulation capabilities and network effects modeling.

## Overview

This model provides a framework for valuing cryptocurrency protocols based on:
- Fee generation and capture
- Network effects
- Market share dynamics
- Growth projections
- Risk-adjusted returns

The implementation includes sensitivity analysis and Monte Carlo simulation to account for the high uncertainty in crypto protocol valuation.

## Features

- **DCF-Based Valuation**: Projects future cash flows with crypto-specific adjustments
- **Network Effects**: Quantifies and incorporates network effect multipliers
- **Monte Carlo Simulation**: Runs multiple scenarios with varying parameters
- **Sensitivity Analysis**: Identifies key value drivers
- **Market Share Constraints**: Incorporates TAM and market share ceilings
- **Risk Adjustments**: Includes crypto-specific risk factors in WACC calculation

## Project Structure

```
├── valuation_base.py       # Core data structures and base functionality
├── valuation_dcf.py        # DCF and projection methods
├── valuation_monte_carlo.py # Monte Carlo simulation capabilities
└── sample_data/
    └── ena_q.csv          # Sample quarterly financial data
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-valuation-model.git

# Install required packages
pip install pandas numpy scipy
```

## Usage

### Basic Valuation

```python
from valuation_base import EnhancedCryptoValuation

# Initialize model
model = EnhancedCryptoValuation("sample_data/ena_q.csv")
model.set_tam(150e9)  # Set TAM to $150 billion

# Run basic valuation
valuation = model.dcf_valuation(
    projection_years=5,
    terminal_growth=0.15,
    fee_capture_rate=0.90,
    risk_free_rate=0.04,
    market_risk_premium=0.06,
    crypto_beta=1.5,
    market_share_ceiling=0.30
)

print(f"Token Price: ${valuation['token_price']:,.2f}")
```

### Monte Carlo Simulation

```python
from valuation_monte_carlo import MonteCarloValuation, SimulationParameters

# Define parameter ranges
sim_params = SimulationParameters(
    tam=(100e9, 200e9),
    terminal_growth=(0.15, 0.25),
    fee_capture_rate=(0.80, 1.00),
    risk_free_rate=(0.03, 0.05),
    market_risk_premium=(0.06, 0.15),
    crypto_beta=(1.3, 1.9),
    market_share_ceiling=(0.25, 0.35)
)

# Run Monte Carlo simulation
mc_model = MonteCarloValuation(model)
results = mc_model.run_monte_carlo(
    params=sim_params,
    num_simulations=1000
)
```

## Input Data Format

The model expects quarterly financial data in CSV format with the following structure:

```csv
Financial statement,Q4 2024,Q3 2024,Q2 2024,Q1 2024,Q4 2023
Outstanding supply,2566289323.27,3034738828.61,2723470253.15,502295177.48,23138306.82
Fees,14566030.78,27741907.18,54137568.33,31920315.05,86033.68
Revenue,14566030.78,27741907.18,54137568.33,31920315.05,86033.68
...
```

Required metrics include:
- Fees
- Revenue
- Outstanding supply
- Circulating supply
- Active users (daily/weekly/monthly)
- Core developers
- Code commits

## Model Parameters

### Key Parameters

- `tam`: Total Addressable Market size
- `terminal_growth`: Long-term growth rate
- `fee_capture_rate`: Proportion of fees captured as revenue
- `market_share_ceiling`: Maximum attainable market share
- `crypto_beta`: Systematic risk factor

### Network Effects Multiplier

The model calculates a network effects multiplier based on:
- User growth rates
- User retention
- Token distribution
- Development activity

## Limitations and Assumptions

1. **Growth Assumptions**: The model assumes protocols can maintain high growth rates initially, transitioning to more sustainable levels.

2. **Market Efficiency**: Assumes some market inefficiencies in crypto markets that create valuation opportunities.

3. **Network Effects**: Assumes network effects follow predictable patterns based on user and developer metrics.

4. **Competition**: Does not explicitly model competitive dynamics beyond market share constraints.

## Contributing

Contributions are welcome! Please submit pull requests for:
- Additional valuation metrics
- Enhanced risk modeling
- Improved network effects calculations
- Better parameter estimation techniques
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on traditional DCF methodology with crypto-specific adaptations
- Incorporates network effects research from crypto economics literature
- Uses market standard approaches for risk adjustment and Monte Carlo simulation
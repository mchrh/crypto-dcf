import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class EnhancedCryptoValuation:
    def __init__(self, csv_path: str):
        """
        Initialize the enhanced valuation model with historical data
        
        Parameters:
        csv_path (str): Path to the CSV file containing protocol metrics
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df.set_index('Financial statement')
        self.df = self.df.apply(pd.to_numeric, errors='ignore')
        self.df = self.df[sorted(self.df.columns, reverse=True)]
        
        self.calculate_derived_metrics()
    
    def calculate_derived_metrics(self):
        """Calculate additional protocol health and efficiency metrics"""
        try:
            # User engagement metrics
            self.df.loc['User retention'] = (
                self.df.loc['Active users (monthly)'] / 
                self.df.loc['Active users (monthly)'].shift(-1)
            ).fillna(0)
            
            # Protocol efficiency metrics
            self.df.loc['Revenue per user'] = (
                self.df.loc['Revenue'] / 
                self.df.loc['Active users (monthly)']
            ).fillna(0)
            
            # Development activity score
            self.df.loc['Dev activity score'] = (
                (self.df.loc['Core developers'] * 0.7) + 
                (self.df.loc['Code commits'] * 0.3)
            ).fillna(0)
            
            # Token velocity
            self.df.loc['Token velocity'] = (
                self.df.loc['Token trading volume'] / 
                self.df.loc['Market cap (circulating)']
            ).fillna(0)
            
        except Exception as e:
            print(f"Error calculating derived metrics: {e}")
    
    def calculate_growth_metrics(self, metric: str, periods: int = 4) -> Dict[str, float]:
        """
        Calculate enhanced growth metrics with statistical confidence
        
        Parameters:
        metric (str): The metric to analyze
        periods (int): Number of periods to analyze
        
        Returns:
        Dict containing various growth metrics and statistical measures
        """
        values = self.df.loc[metric].dropna()
        if len(values) < periods:
            return {
                'cagr': np.nan,
                'avg_growth': np.nan,
                'recent_growth': np.nan,
                'growth_volatility': np.nan,
                'confidence_interval': (np.nan, np.nan)
            }
        
        growth_rates = values.pct_change(-1).dropna()
        
        # Calculate CAGR with volatility adjustment
        cagr = (values.iloc[0] / values.iloc[periods-1]) ** (1/periods) - 1
        volatility = growth_rates.std()
        
        # Calculate confidence intervals
        if len(growth_rates) > 1:
            confidence = 0.95
            degrees_of_freedom = len(growth_rates) - 1
            mean = growth_rates.mean()
            standard_error = stats.sem(growth_rates)
            conf_interval = stats.t.interval(confidence=confidence,
                                          df=degrees_of_freedom,
                                          loc=mean,
                                          scale=standard_error)
        else:
            conf_interval = (np.nan, np.nan)
        
        return {
            'cagr': cagr,
            'avg_growth': growth_rates.mean(),
            'recent_growth': growth_rates.iloc[0],
            'growth_volatility': volatility,
            'confidence_interval': conf_interval
        }
    
    def estimate_market_size(self, 
                           base_market_size: float,
                           cagr: float,
                           years: int) -> float:
        """
        Estimate total addressable market (TAM) size
        
        Parameters:
        base_market_size (float): Current market size
        cagr (float): Expected market CAGR
        years (int): Projection period
        
        Returns:
        float: Estimated market size
        """
        return 150e9 #base_market_size * (1 + cagr) ** years
    
    def calculate_network_effects_multiplier(self) -> float:
        """
        Calculate a multiplier based on network effects strength
        
        Returns:
        float: Network effects multiplier
        """
        recent_user_growth = self.calculate_growth_metrics('Active users (monthly)')
        user_retention = self.df.loc['User retention'].mean()
        token_distribution = self.df.loc['Tokenholders'].iloc[0] / self.df.loc['Circulating supply'].iloc[0]
        
        multiplier = (
            (1 + recent_user_growth['cagr']) * 0.4 +
            user_retention * 0.3 +
            min(token_distribution * 100, 1) * 0.3
        )
        
        return max(0.8, min(multiplier, 1.5))  # Cap multiplier between 0.8 and 1.5
    
    def calculate_risk_adjusted_wacc(self,
                                   risk_free_rate: float = 0.04,
                                   market_risk_premium: float = 0.06,
                                   crypto_beta: float = 1.5) -> float:
        """
        Calculate risk-adjusted WACC incorporating protocol-specific factors
        
        Parameters:
        risk_free_rate (float): Risk-free rate
        market_risk_premium (float): Market risk premium
        crypto_beta (float): Base crypto beta
        
        Returns:
        float: Adjusted WACC
        """
        dev_activity = self.df.loc['Dev activity score'].mean()
        token_concentration = 1 - (self.df.loc['Token turnover (circulating)'].mean())
        user_growth_volatility = self.calculate_growth_metrics('Active users (monthly)')['growth_volatility']
        
        adjusted_beta = crypto_beta * (
            1 +
            (0.2 * (1 - dev_activity)) +  # Higher risk for low development activity
            (0.15 * token_concentration) +  # Higher risk for concentrated token holdings
            (0.15 * user_growth_volatility)  # Higher risk for volatile user growth
        )
        
        return risk_free_rate + (market_risk_premium * adjusted_beta)
    
    def project_future_fees(self,
                          projection_years: int = 5,
                          terminal_growth: float = 0.02,
                          market_share_ceiling: float = 0.30,
                          min_growth_rate: float = 0.30) -> Tuple[pd.Series, Dict]:
        """
        Project future protocol fees with market share constraints and network effects
        
        Parameters:
        projection_years (int): Number of years to project
        terminal_growth (float): Long-term growth rate assumption
        market_share_ceiling (float): Maximum market share assumption
        min_growth_rate (float): Minimum annual growth rate
        
        Returns:
        Tuple[pd.Series, Dict]: Projected fees and projection metrics
        """
        historical_fees = self.df.loc['Fees'].dropna()
        growth_metrics = self.calculate_growth_metrics('Fees')
        
        # Ensure we have positive historical fees for calculation
        latest_fee = max(historical_fees.iloc[0], historical_fees.mean())
        if latest_fee <= 0:
            latest_fee = historical_fees.max()  # Use maximum if recent fees are negative/zero
        
        # Adjust growth expectations based on network effects
        network_multiplier = self.calculate_network_effects_multiplier()
        
        # Set higher base growth rate expectations
        base_growth = max(
            growth_metrics['cagr'] if not np.isnan(growth_metrics['cagr']) else 0.35,
            min_growth_rate
        )
        initial_growth = min(max(base_growth * network_multiplier, min_growth_rate), 0.75)  # Cap growth between min_growth_rate and 75%
        
        # Consider market share constraints with floor
        total_market_cap = sum(self.df.loc['Market cap (circulating)'].dropna())
        current_market_share = min(
            max(
                self.df.loc['Market cap (circulating)'].iloc[0] / total_market_cap 
                if total_market_cap > 0 else 0.01,
                0.01
            ),
            market_share_ceiling
        )
        
        terminal_growth = min(max(terminal_growth, min_growth_rate/2), min_growth_rate)  # Between min_growth_rate/2 and min_growth_rate
        growth_decay = max((initial_growth - terminal_growth) / projection_years, 0)
        
        projected_fees = []
        projection_metrics = {
            'network_multiplier': network_multiplier,
            'market_share': [],
            'growth_rates': []
        }
        
        last_fee = latest_fee
        
        for year in range(projection_years):
            projected_market_share = min(
                current_market_share * (1 + initial_growth) ** year,
                market_share_ceiling
            )
            
            # Ensure growth rate stays above minimum threshold
            growth_rate = max(
                min(
                    initial_growth - (growth_decay * year),
                    (market_share_ceiling - current_market_share) / current_market_share
                ),
                min_growth_rate  # Enforce minimum growth rate
            )
            
            projected_fee = last_fee * (1 + growth_rate)
            projected_fees.append(projected_fee)
            
            projection_metrics['market_share'].append(projected_market_share)
            projection_metrics['growth_rates'].append(growth_rate)
            
            last_fee = projected_fee
        
        #return pd.Series(projected_fees), projection_metrics
        
        projected_fees = []
        projection_metrics = {
            'network_multiplier': network_multiplier,
            'market_share': [],
            'growth_rates': []
        }
        
        last_fee = historical_fees.iloc[0]
        
        for year in range(projection_years):
            # Calculate market share-constrained growth
            projected_market_share = min(
                current_market_share * (1 + initial_growth) ** year,
                market_share_ceiling
            )
            
            growth_rate = max(
                min(
                    initial_growth - (growth_decay * year),
                    (market_share_ceiling - current_market_share) / current_market_share
                ),
                terminal_growth
            )
            
            projected_fee = last_fee * (1 + growth_rate)
            projected_fees.append(projected_fee)
            
            projection_metrics['market_share'].append(projected_market_share)
            projection_metrics['growth_rates'].append(growth_rate)
            
            last_fee = projected_fee
        
        return pd.Series(projected_fees), projection_metrics
    
    def dcf_valuation(self,
                     projection_years: int = 5,
                     terminal_growth: float = 0.02,
                     fee_capture_rate: float = 0.167,
                     risk_free_rate: float = 0.04,
                     market_risk_premium: float = 0.06,
                     crypto_beta: float = 1.5,
                     market_share_ceiling: float = 0.30) -> Dict[str, float]:
        """
        Perform enhanced DCF valuation incorporating network effects and risk adjustments
        
        Parameters:
        [previous parameters]
        market_share_ceiling (float): Maximum market share assumption
        
        Returns:
        Dict containing detailed valuation metrics
        """
        # Calculate risk-adjusted WACC
        wacc = self.calculate_risk_adjusted_wacc(
            risk_free_rate,
            market_risk_premium,
            crypto_beta
        )
        
        # Project future fees with network effects
        projected_fees, projection_metrics = self.project_future_fees(
            projection_years,
            terminal_growth,
            market_share_ceiling
        )
        
        protocol_revenue = projected_fees * fee_capture_rate
        
        # Calculate present values with floor
        discount_factors = pd.Series([(1 + wacc) ** -i for i in range(1, projection_years + 1)])
        pv_cash_flows = protocol_revenue * discount_factors
        
        # Ensure terminal growth is less than WACC to avoid negative denominator
        safe_terminal_growth = min(terminal_growth, wacc - 0.01)
        
        # Terminal value with network effects adjustment and sanity checks
        terminal_value = max(
            protocol_revenue.iloc[-1] * 
            (1 + safe_terminal_growth) * 
            projection_metrics['network_multiplier'] / 
            (wacc - safe_terminal_growth),
            0  # Ensure non-negative terminal value
        )
        
        pv_terminal_value = terminal_value * discount_factors.iloc[-1]
        
        # Apply minimum value constraint
        pv_terminal_value = max(pv_terminal_value, 0)
        
        # Calculate enterprise value
        enterprise_value = pv_cash_flows.sum() + pv_terminal_value
        
        # Calculate per-token metrics
        circulating_supply = self.df.loc['Circulating supply'].iloc[0]
        current_price = self.df.loc['Price'].iloc[0]
        
        return {
            'enterprise_value': enterprise_value,
            'token_price': enterprise_value / circulating_supply,
            'current_price': current_price,
            'price_upside': (enterprise_value / circulating_supply / current_price - 1) if current_price else np.nan,
            'wacc': wacc,
            'terminal_value': terminal_value,
            'pv_cash_flows': pv_cash_flows.sum(),
            'pv_terminal_value': pv_terminal_value,
            'network_multiplier': projection_metrics['network_multiplier'],
            'projected_market_shares': projection_metrics['market_share'],
            'growth_rates': projection_metrics['growth_rates']
        }


if __name__ == "__main__":
    token_name = 'uni'
    model = EnhancedCryptoValuation(f"statements/{token_name}_q.csv")
    
    # Basic valuation
    valuation = model.dcf_valuation(
        projection_years=10,
        terminal_growth=0.5,
        fee_capture_rate=0.167,
        risk_free_rate=0.02,
        market_risk_premium=0.1,
        crypto_beta=1.25,
        market_share_ceiling=0.30
    )
    
    print(f"\nValuation Results for {token_name.upper()}-USD:")
    for key, value in valuation.items():
        if isinstance(value, float):
            print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value}")
    
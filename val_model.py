import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings 

warnings.filterwarnings('ignore')

class CryptoValuation:
    def __init__(self, csv_path: str):
        """
        Initialize the valuation model with historical data
        
        Parameters:
        csv_path (str): Path to the CSV file containing protocol metrics
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df.set_index('Financial statement')
        self.df = self.df.apply(pd.to_numeric, errors='ignore')
        
        self.df = self.df[sorted(self.df.columns, reverse=True)]
        
    def calculate_growth_metrics(self, metric: str, periods: int = 4) -> Dict[str, float]:
        """
        Calculate growth rates for a given metric
        
        Parameters:
        metric (str): The metric to analyze (e.g., 'Fees', 'Trading volume')
        periods (int): Number of periods to analyze
        
        Returns:
        Dict containing various growth metrics
        """
        values = self.df.loc[metric].dropna()
        if len(values) < periods:
            return {
                'cagr': np.nan,
                'avg_growth': np.nan,
                'recent_growth': np.nan
            }
        
        growth_rates = values.pct_change(-1).dropna()
        
        cagr = (values.iloc[0] / values.iloc[periods-1]) ** (1/periods) - 1
        
        return {
            'cagr': cagr,
            'avg_growth': growth_rates.mean(),
            'recent_growth': growth_rates.iloc[0]
        }
    
    def project_future_fees(self, 
                          projection_years: int = 5,
                          terminal_growth: float = 0.02) -> pd.Series:
        """
        Project future protocol fees based on historical data and growth assumptions
        
        Parameters:
        projection_years (int): Number of years to project
        terminal_growth (float): Long-term growth rate assumption
        
        Returns:
        pd.Series: Projected fees for each future period
        """
        historical_fees = self.df.loc['Fees'].dropna()
        
        growth_metrics = self.calculate_growth_metrics('Fees')
        
        initial_growth = growth_metrics['cagr']
        growth_decay = (initial_growth - terminal_growth) / projection_years
        
        projected_fees = []
        last_fee = historical_fees.iloc[0]
        
        for year in range(projection_years):
            growth_rate = max(initial_growth - (growth_decay * year), terminal_growth)
            projected_fee = last_fee * (1 + growth_rate)
            projected_fees.append(projected_fee)
            last_fee = projected_fee
            
        return pd.Series(projected_fees)
    
    def calculate_wacc(self,
                      risk_free_rate: float = 0.04,
                      market_risk_premium: float = 0.06,
                      crypto_beta: float = 1.5) -> float:
        """
        Calculate weighted average cost of capital adapted for crypto
        
        Parameters:
        risk_free_rate (float): Risk-free rate
        market_risk_premium (float): Market risk premium
        crypto_beta (float): Beta for the crypto protocol
        
        Returns:
        float: Calculated WACC
        """
        return risk_free_rate + (market_risk_premium * crypto_beta)
    
    def dcf_valuation(self,
                     projection_years: int = 5,
                     terminal_growth: float = 0.02,
                     fee_capture_rate: float = 0.167, # Assuming protocol captures 1/6 of fees
                     risk_free_rate: float = 0.04,
                     market_risk_premium: float = 0.06,
                     crypto_beta: float = 1.5) -> Dict[str, float]:
        """
        Perform DCF valuation for the protocol
        
        Parameters:
        projection_years (int): Number of years to project
        terminal_growth (float): Long-term growth rate
        fee_capture_rate (float): Portion of fees captured by token holders
        risk_free_rate (float): Risk-free rate
        market_risk_premium (float): Market risk premium
        crypto_beta (float): Beta for the crypto protocol
        
        Returns:
        Dict containing valuation metrics
        """
        wacc = self.calculate_wacc(risk_free_rate, market_risk_premium, crypto_beta)
        
        projected_fees = self.project_future_fees(projection_years, terminal_growth)
        
        protocol_revenue = projected_fees * fee_capture_rate
        
        discount_factors = pd.Series([(1 + wacc) ** -i for i in range(1, projection_years + 1)])
        pv_cash_flows = protocol_revenue * discount_factors
        
        terminal_value = (protocol_revenue.iloc[-1] * (1 + terminal_growth) / 
                         (wacc - terminal_growth))
        pv_terminal_value = terminal_value * discount_factors.iloc[-1]
        
        enterprise_value = pv_cash_flows.sum() + pv_terminal_value
        
        circulating_supply = self.df.loc['Circulating supply'].iloc[0]
        
        return {
            'enterprise_value': enterprise_value,
            'token_price': enterprise_value / circulating_supply,
            'current_price': self.df.loc['Price'].iloc[0],
            'wacc': wacc,
            'terminal_value': terminal_value,
            'pv_cash_flows': pv_cash_flows.sum(),
            'pv_terminal_value': pv_terminal_value
        }
    
    def sensitivity_analysis(self,
                           terminal_growth_range: List[float] = None,
                           wacc_range: List[float] = None) -> pd.DataFrame:
        """
        Perform sensitivity analysis on key valuation inputs
        
        Parameters:
        terminal_growth_range (List[float]): Range of terminal growth rates to test
        wacc_range (List[float]): Range of WACC values to test
        
        Returns:
        pd.DataFrame: Matrix of valuation outcomes
        """
        if terminal_growth_range is None:
            terminal_growth_range = [0.05, 0.06, 0.07, 0.08, 0.1]
        if wacc_range is None:
            wacc_range = [0.08, 0.10, 0.12, 0.14, 0.16]
        
        results = []
        for growth in terminal_growth_range:
            row = []
            for wacc in wacc_range:
                valuation = self.dcf_valuation(
                    terminal_growth=growth,
                    risk_free_rate=wacc-0.06  
                )
                row.append(valuation['token_price'])
            results.append(row)
        
        return pd.DataFrame(
            results,
            index=[f'{g*100:.1f}%' for g in terminal_growth_range],
            columns=[f'{w*100:.1f}%' for w in wacc_range]
        )

if __name__ == "__main__":

    token_name = 'ena'

    model = CryptoValuation(f"statements/{token_name}_q.csv")

    valuation = model.dcf_valuation(
        projection_years=5,
        terminal_growth=0.2,
        fee_capture_rate=0.167,
        risk_free_rate=0.2,
        market_risk_premium=0.1,
        crypto_beta=1.25
    )
    
    #valuation = model.dcf_valuation()
    print(f"\nValuation Result for {token_name.upper()}-USD:")
    for key, value in valuation.items():
        print(f"{key}: ${value:,.2f}")
    
    sensitivity = model.sensitivity_analysis()
    print("\nSensitivity Analysis (Token Price):")
    print(sensitivity)
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats
from dataclasses import dataclass
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt
from dataclasses import asdict
import seaborn as sns

warnings.filterwarnings('ignore')

@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation"""
    tam: Tuple[float, float]  # (min, max) for triangular distribution
    terminal_growth: Tuple[float, float]  # (min, max)
    fee_capture_rate: Tuple[float, float]  # (min, max)
    risk_free_rate: Tuple[float, float]  # (min, max)
    market_risk_premium: Tuple[float, float]  # (min, max)
    crypto_beta: Tuple[float, float]  # (min, max)
    market_share_ceiling: Tuple[float, float]  # (min, max)

class EnhancedCryptoValuation:
    def __init__(self, csv_path: str, tam: Optional[float] = None):
        """
        Initialize the enhanced valuation model with historical data and optional TAM
        
        Parameters:
        csv_path (str): Path to the CSV file containing protocol metrics
        tam (float, optional): Total Addressable Market size in USD
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df.set_index('Financial statement')
        self.df = self.df.apply(pd.to_numeric, errors='ignore')
        self.df = self.df[sorted(self.df.columns, reverse=True)]
        self.tam = tam
        
        self.calculate_derived_metrics()
    
    def set_tam(self, tam: float):
        """Set Total Addressable Market size"""
        self.tam = tam

    def get_market_size(self) -> float:
        """Get current TAM value"""
        if self.tam is None:
            raise ValueError("TAM has not been set. Please set TAM using set_tam() method.")
        return self.tam
    
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
        if self.tam is None:
            raise ValueError("TAM must be set before projecting future fees")
            
        historical_fees = self.df.loc['Fees'].dropna()
        growth_metrics = self.calculate_growth_metrics('Fees')
        
        # Calculate current market penetration
        latest_fee = max(historical_fees.iloc[0], historical_fees.mean())
        current_penetration = latest_fee / self.tam if self.tam > 0 else 0
        
        # Adjust growth expectations based on market penetration and network effects
        network_multiplier = self.calculate_network_effects_multiplier()
        base_growth = max(
            growth_metrics['cagr'] if not np.isnan(growth_metrics['cagr']) else 0.35,
            min_growth_rate
        )
        
        # Reduce growth as we approach TAM
        penetration_adjustment = 1 - (current_penetration / market_share_ceiling)
        initial_growth = min(
            max(base_growth * network_multiplier * penetration_adjustment, min_growth_rate),
            0.75
        )
        
        terminal_growth = min(max(terminal_growth, min_growth_rate/2), min_growth_rate)
        growth_decay = max((initial_growth - terminal_growth) / projection_years, 0)
        
        projected_fees = []
        projection_metrics = {
            'network_multiplier': network_multiplier,
            'market_share': [],
            'growth_rates': []
        }
        
        last_fee = latest_fee
        current_market_share = current_penetration
        
        for year in range(projection_years):
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
            
            projected_fee = min(last_fee * (1 + growth_rate), self.tam * market_share_ceiling)
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
        projection_years (int): Years to project
        terminal_growth (float): Long-term growth rate
        fee_capture_rate (float): Protocol's fee capture rate
        risk_free_rate (float): Risk-free rate
        market_risk_premium (float): Market risk premium
        crypto_beta (float): Base crypto beta
        market_share_ceiling (float): Maximum market share
        
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
        
        # Ensure terminal growth is less than WACC
        safe_terminal_growth = min(terminal_growth, wacc - 0.01)
        
        # Terminal value with network effects adjustment
        terminal_value = max(
            protocol_revenue.iloc[-1] * 
            (1 + safe_terminal_growth) * 
            projection_metrics['network_multiplier'] / 
            (wacc - safe_terminal_growth),
            0
        )
        
        pv_terminal_value = terminal_value * discount_factors.iloc[-1]
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
    
class MonteCarloValuation:
    def __init__(self, base_model: 'EnhancedCryptoValuation'):
        """
        Initialize Monte Carlo simulation wrapper
        
        Parameters:
        base_model: Instance of EnhancedCryptoValuation
        """
        self.model = base_model
        
    def run_single_simulation(self, params: SimulationParameters) -> Optional[Dict]:
        """
        Run a single simulation with sampled parameters
        
        Parameters:
        params: SimulationParameters object with parameter ranges
        
        Returns:
        Dict containing valuation results or None if simulation fails
        """
        try:
            # Sample parameters from distributions
            tam = np.random.triangular(
                params.tam[0],
                np.mean(params.tam),
                params.tam[1]
            )
            
            term_growth = np.random.uniform(*params.terminal_growth)
            fee_capture = np.random.uniform(*params.fee_capture_rate)
            rf_rate = np.random.uniform(*params.risk_free_rate)
            mrkt_premium = np.random.uniform(*params.market_risk_premium)
            beta = np.random.uniform(*params.crypto_beta)
            market_share = np.random.uniform(*params.market_share_ceiling)
            
            # Set TAM for this simulation
            self.model.set_tam(tam)
            
            # Run valuation with sampled parameters
            valuation = self.model.dcf_valuation(
                terminal_growth=term_growth,
                fee_capture_rate=fee_capture,
                risk_free_rate=rf_rate,
                market_risk_premium=mrkt_premium,
                crypto_beta=beta,
                market_share_ceiling=market_share
            )
            
            # Add sampled parameters to result
            valuation.update({
                'sampled_tam': tam,
                'sampled_terminal_growth': term_growth,
                'sampled_fee_capture': fee_capture,
                'sampled_rf_rate': rf_rate,
                'sampled_market_premium': mrkt_premium,
                'sampled_beta': beta,
                'sampled_market_share': market_share
            })
            
            return valuation
            
        except Exception as e:
            print(f"Simulation failed with error: {e}")
            return None

    def run_monte_carlo(self,
                       params: SimulationParameters,
                       num_simulations: int = 1000,
                       projection_years: int = 5,
                       num_processes: Optional[int] = None) -> Dict:
        """
        Run Monte Carlo simulation of the valuation model
        
        Parameters:
        params: SimulationParameters object containing parameter ranges
        num_simulations: Number of simulations to run
        projection_years: Number of years to project
        num_processes: Number of processes for parallel execution
        
        Returns:
        Dict containing simulation results and statistics
        """
        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() - 1)
        
        # Run simulations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.run_single_simulation, params) 
                      for _ in range(num_simulations)]
            results = [f.result() for f in futures if f.result() is not None]
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        stats_dict = {
            'token_price_stats': {
                'mean': results_df['token_price'].mean(),
                'median': results_df['token_price'].median(),
                'std': results_df['token_price'].std(),
                'percentiles': {
                    '5%': results_df['token_price'].quantile(0.05),
                    '25%': results_df['token_price'].quantile(0.25),
                    '75%': results_df['token_price'].quantile(0.75),
                    '95%': results_df['token_price'].quantile(0.95)
                }
            },
            'ev_stats': {
                'mean': results_df['enterprise_value'].mean(),
                'median': results_df['enterprise_value'].median(),
                'std': results_df['enterprise_value'].std(),
                'percentiles': {
                    '5%': results_df['enterprise_value'].quantile(0.05),
                    '25%': results_df['enterprise_value'].quantile(0.25),
                    '75%': results_df['enterprise_value'].quantile(0.75),
                    '95%': results_df['enterprise_value'].quantile(0.95)
                }
            },
            'simulation_count': len(results),
            'success_rate': len(results) / num_simulations,
            'results_df': results_df  # Store full results for additional analysis
        }
        
        return stats_dict

    def analyze_sensitivity(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze parameter sensitivity based on simulation results
        
        Parameters:
        results_df: DataFrame containing simulation results
        
        Returns:
        Dict containing correlation analysis and parameter impacts
        """
        # Parameters to analyze
        param_cols = [col for col in results_df.columns if col.startswith('sampled_')]
        
        # Calculate correlations with token price
        correlations = {}
        for param in param_cols:
            corr = results_df[param].corr(results_df['token_price'])
            correlations[param.replace('sampled_', '')] = corr
        
        # Rank parameters by absolute correlation
        ranked_params = dict(sorted(correlations.items(), 
                                  key=lambda x: abs(x[1]), 
                                  reverse=True))
        
        return {
            'correlations': correlations,
            'ranked_impacts': ranked_params
        }

def run_valuation_analysis(csv_path: str,
                          params: SimulationParameters,
                          num_simulations: int = 1000) -> None:
    """
    Run complete valuation analysis including Monte Carlo simulation
    
    Parameters:
    csv_path: Path to the CSV file with financial data
    params: SimulationParameters object with parameter ranges
    num_simulations: Number of simulations to run
    """
    base_model = EnhancedCryptoValuation(csv_path)
    base_model.set_tam(np.mean(params.tam))
    
    base_case = base_model.dcf_valuation()
    
    mc_model = MonteCarloValuation(base_model)
    
    simulation_results = mc_model.run_monte_carlo(
        params=params,
        num_simulations=num_simulations
    )
    
    sensitivity = mc_model.analyze_sensitivity(simulation_results['results_df'])
    
    print("\nBase Case Valuation:")
    print(f"Token Price: ${base_case['token_price']:,.2f}")
    print(f"Enterprise Value: ${base_case['enterprise_value']:,.2f}")
    
    print("\nMonte Carlo Simulation Results:")
    print("\nToken Price Statistics:")
    for key, value in simulation_results['token_price_stats'].items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for pct, val in value.items():
                print(f"  {pct}: ${val:,.2f}")
        else:
            print(f"{key}: ${value:,.2f}")
    
    print("\nParameter Sensitivity (correlation with token price):")
    for param, impact in sensitivity['ranked_impacts'].items():
        print(f"{param}: {impact:.3f}")
    
    print(f"\nSimulation Success Rate: {simulation_results['success_rate']*100:.1f}%")


if __name__ == "__main__":

    token_name = "ena"

    params = SimulationParameters(
        tam=(180e9, 300e9),  
        terminal_growth=(0.5, 0.75),
        fee_capture_rate=(0.05, 0.15),
        risk_free_rate=(0.0, 0.03),
        market_risk_premium=(0.10, 0.20),
        crypto_beta=(1.2, 1.8),
        market_share_ceiling=(0.10, 0.25)
    )
    
    print(f"\n{token_name.upper()}-USD")
    run_valuation_analysis(
        csv_path=f"statements/{token_name}_q.csv",
        params=params,
        num_simulations=2500
    )
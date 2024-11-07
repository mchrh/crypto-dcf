import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from src.val import EnhancedCryptoValuation
from src.params import SimulationParameters

warnings.filterwarnings('ignore')



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


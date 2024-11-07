import numpy as np 
from src.mc import MonteCarloValuation
from src.val import EnhancedCryptoValuation
from src.params import SimulationParameters


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
        terminal_growth=(0.75, 1),
        fee_capture_rate=(0.05, 0.25),
        risk_free_rate=(0.0, 0.03),
        market_risk_premium=(0.15, 0.20),
        crypto_beta=(1.2, 1.8),
        market_share_ceiling=(0.10, 0.25)
    )
    
    print(f"\n{token_name.upper()}-USD")
    run_valuation_analysis(
        csv_path=f"statements/{token_name}_q.csv",
        params=params,
        num_simulations=10_000)
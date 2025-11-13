"""
Monte Carlo Simulation Module for Geopolitical Analysis
Provides probabilistic modeling for various geopolitical scenarios
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of geopolitical scenarios that can be modeled"""
    MILITARY_ATTRITION = "military_attrition"
    ECONOMIC_SANCTIONS = "economic_sanctions"
    DIPLOMATIC_CRISIS = "diplomatic_crisis"
    RESOURCE_DEPLETION = "resource_depletion"
    ALLIANCE_STABILITY = "alliance_stability"
    CONFLICT_ESCALATION = "conflict_escalation"


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results"""
    scenario_type: str
    iterations: int
    mean: float
    median: float
    std_dev: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    confidence_interval_95: Tuple[float, float]
    probability_thresholds: Dict[str, float]
    timeline: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class MonteCarloSimulator:
    """
    Monte Carlo simulator for geopolitical analysis scenarios
    Handles uncertainty and probabilistic outcomes
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the Monte Carlo simulator
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info("Monte Carlo Simulator initialized")
    
    def simulate_military_attrition(
        self,
        initial_stock: float,
        production_rate: float,
        loss_rate: float,
        time_periods: int,
        iterations: int = 10000,
        production_uncertainty: float = 0.15,
        loss_uncertainty: float = 0.20,
        shock_probability: float = 0.05,
        shock_magnitude: float = 0.30
    ) -> SimulationResult:
        """
        Simulate military equipment attrition with uncertainty
        
        Args:
            initial_stock: Starting equipment count
            production_rate: Annual production rate
            loss_rate: Annual loss rate (as decimal, e.g., 0.15 for 15%)
            time_periods: Number of years to simulate
            iterations: Number of Monte Carlo iterations
            production_uncertainty: Std dev of production rate (as % of mean)
            loss_uncertainty: Std dev of loss rate (as % of mean)
            shock_probability: Probability of major disruption per period
            shock_magnitude: Magnitude of shock when it occurs (as % loss)
        
        Returns:
            SimulationResult with timeline projections
        """
        logger.info(f"Running military attrition simulation: {iterations} iterations over {time_periods} periods")
        
        final_stocks = []
        all_timelines = []
        
        for _ in range(iterations):
            stock = initial_stock
            timeline = [stock]
            
            for period in range(time_periods):
                # Sample production and loss with uncertainty
                production = np.random.normal(
                    production_rate,
                    production_rate * production_uncertainty
                )
                production = max(0, production)  # Can't be negative
                
                # Loss rate with uncertainty
                current_loss_rate = np.random.normal(
                    loss_rate,
                    loss_rate * loss_uncertainty
                )
                current_loss_rate = max(0, min(1, current_loss_rate))  # Clamp between 0 and 1
                
                # Calculate losses
                losses = stock * current_loss_rate
                
                # Apply shock events (sanctions, major battles, etc.)
                if np.random.random() < shock_probability:
                    shock_loss = stock * shock_magnitude
                    losses += shock_loss
                
                # Update stock
                stock = stock + production - losses
                stock = max(0, stock)  # Can't go negative
                
                timeline.append(stock)
            
            final_stocks.append(stock)
            all_timelines.append(timeline)
        
        # Calculate statistics
        final_stocks = np.array(final_stocks)
        mean_timeline = np.mean(all_timelines, axis=0)
        
        result = SimulationResult(
            scenario_type=ScenarioType.MILITARY_ATTRITION.value,
            iterations=iterations,
            mean=np.mean(final_stocks),
            median=np.median(final_stocks),
            std_dev=np.std(final_stocks),
            percentile_5=np.percentile(final_stocks, 5),
            percentile_25=np.percentile(final_stocks, 25),
            percentile_75=np.percentile(final_stocks, 75),
            percentile_95=np.percentile(final_stocks, 95),
            confidence_interval_95=(
                np.percentile(final_stocks, 2.5),
                np.percentile(final_stocks, 97.5)
            ),
            probability_thresholds={
                "critical_depletion_below_500": np.mean(final_stocks < 500),
                "severe_depletion_below_1000": np.mean(final_stocks < 1000),
                "moderate_depletion_below_2000": np.mean(final_stocks < 2000),
                "maintained_above_initial": np.mean(final_stocks >= initial_stock)
            },
            timeline=mean_timeline.tolist(),
            metadata={
                "initial_stock": initial_stock,
                "production_rate": production_rate,
                "loss_rate": loss_rate,
                "time_periods": time_periods
            }
        )
        
        logger.info(f"Simulation complete. Mean final stock: {result.mean:.0f}")
        return result
    
    def simulate_economic_impact(
        self,
        baseline_gdp: float,
        sanctions_impact: float,
        adaptation_rate: float,
        time_periods: int,
        iterations: int = 10000,
        impact_uncertainty: float = 0.25,
        external_shock_prob: float = 0.10
    ) -> SimulationResult:
        """
        Simulate economic impact of sanctions or trade disruptions
        
        Args:
            baseline_gdp: Starting GDP (in billions)
            sanctions_impact: Initial sanctions impact (as decimal, e.g., 0.08 for 8% decline)
            adaptation_rate: Rate of economic adaptation per period (recovery)
            time_periods: Number of years to simulate
            iterations: Number of Monte Carlo iterations
            impact_uncertainty: Uncertainty in sanctions impact
            external_shock_prob: Probability of additional external shocks
        
        Returns:
            SimulationResult for GDP trajectory
        """
        logger.info(f"Running economic sanctions simulation: {iterations} iterations")
        
        final_gdps = []
        
        for _ in range(iterations):
            gdp = baseline_gdp
            cumulative_impact = sanctions_impact
            
            for period in range(time_periods):
                # Impact decreases over time as economy adapts
                current_impact = cumulative_impact * (1 - adaptation_rate) ** period
                
                # Add uncertainty
                actual_impact = np.random.normal(
                    current_impact,
                    current_impact * impact_uncertainty
                )
                
                # External shocks (secondary sanctions, supply chain issues)
                if np.random.random() < external_shock_prob:
                    shock = np.random.uniform(0.02, 0.05)
                    actual_impact += shock
                
                # Apply impact to GDP
                gdp = baseline_gdp * (1 - actual_impact)
            
            final_gdps.append(gdp)
        
        final_gdps = np.array(final_gdps)
        
        result = SimulationResult(
            scenario_type=ScenarioType.ECONOMIC_SANCTIONS.value,
            iterations=iterations,
            mean=np.mean(final_gdps),
            median=np.median(final_gdps),
            std_dev=np.std(final_gdps),
            percentile_5=np.percentile(final_gdps, 5),
            percentile_25=np.percentile(final_gdps, 25),
            percentile_75=np.percentile(final_gdps, 75),
            percentile_95=np.percentile(final_gdps, 95),
            confidence_interval_95=(
                np.percentile(final_gdps, 2.5),
                np.percentile(final_gdps, 97.5)
            ),
            probability_thresholds={
                "severe_contraction_below_80pct": np.mean(final_gdps < baseline_gdp * 0.80),
                "moderate_contraction_below_90pct": np.mean(final_gdps < baseline_gdp * 0.90),
                "mild_contraction_below_95pct": np.mean(final_gdps < baseline_gdp * 0.95)
            },
            metadata={
                "baseline_gdp": baseline_gdp,
                "sanctions_impact": sanctions_impact,
                "adaptation_rate": adaptation_rate,
                "time_periods": time_periods
            }
        )
        
        logger.info(f"Economic simulation complete. Mean GDP impact: {((result.mean / baseline_gdp - 1) * 100):.1f}%")
        return result
    
    def simulate_conflict_escalation(
        self,
        initial_intensity: float,
        escalation_factors: List[float],
        de_escalation_factors: List[float],
        time_periods: int,
        iterations: int = 10000,
        diplomatic_intervention_prob: float = 0.15,
        major_incident_prob: float = 0.08
    ) -> SimulationResult:
        """
        Simulate conflict escalation dynamics with multiple factors
        
        Args:
            initial_intensity: Starting conflict intensity (0-10 scale)
            escalation_factors: List of factors that increase intensity
            de_escalation_factors: List of factors that decrease intensity
            time_periods: Number of time periods (e.g., months)
            iterations: Number of Monte Carlo iterations
            diplomatic_intervention_prob: Probability of diplomatic intervention
            major_incident_prob: Probability of major escalatory incident
        
        Returns:
            SimulationResult for conflict intensity trajectory
        """
        logger.info(f"Running conflict escalation simulation: {iterations} iterations")
        
        final_intensities = []
        escalation_count = 0
        
        for _ in range(iterations):
            intensity = initial_intensity
            
            for period in range(time_periods):
                # Base drift (random walk)
                drift = np.random.normal(0, 0.5)
                
                # Apply escalation factors (weighted)
                escalation_effect = np.random.choice(
                    escalation_factors,
                    size=1,
                    p=[f / sum(escalation_factors) for f in escalation_factors]
                )[0]
                
                # Apply de-escalation factors
                de_escalation_effect = np.random.choice(
                    de_escalation_factors,
                    size=1,
                    p=[f / sum(de_escalation_factors) for f in de_escalation_factors]
                )[0]
                
                # Diplomatic intervention
                if np.random.random() < diplomatic_intervention_prob:
                    intensity -= np.random.uniform(0.5, 1.5)
                
                # Major incident
                if np.random.random() < major_incident_prob:
                    intensity += np.random.uniform(1.0, 2.5)
                
                # Update intensity
                intensity = intensity + drift + escalation_effect - de_escalation_effect
                intensity = max(0, min(10, intensity))  # Clamp between 0 and 10
            
            final_intensities.append(intensity)
            if intensity > initial_intensity * 1.5:
                escalation_count += 1
        
        final_intensities = np.array(final_intensities)
        
        result = SimulationResult(
            scenario_type=ScenarioType.CONFLICT_ESCALATION.value,
            iterations=iterations,
            mean=np.mean(final_intensities),
            median=np.median(final_intensities),
            std_dev=np.std(final_intensities),
            percentile_5=np.percentile(final_intensities, 5),
            percentile_25=np.percentile(final_intensities, 25),
            percentile_75=np.percentile(final_intensities, 75),
            percentile_95=np.percentile(final_intensities, 95),
            confidence_interval_95=(
                np.percentile(final_intensities, 2.5),
                np.percentile(final_intensities, 97.5)
            ),
            probability_thresholds={
                "high_intensity_above_7": np.mean(final_intensities > 7),
                "moderate_intensity_4to7": np.mean((final_intensities >= 4) & (final_intensities <= 7)),
                "low_intensity_below_4": np.mean(final_intensities < 4),
                "escalation_from_baseline": escalation_count / iterations
            },
            metadata={
                "initial_intensity": initial_intensity,
                "time_periods": time_periods
            }
        )
        
        logger.info(f"Conflict simulation complete. Escalation probability: {result.probability_thresholds['escalation_from_baseline']:.1%}")
        return result
    
    def simulate_alliance_stability(
        self,
        alliance_strength: float,
        internal_cohesion: float,
        external_pressure: float,
        time_periods: int,
        iterations: int = 10000,
        crisis_probability: float = 0.12
    ) -> SimulationResult:
        """
        Simulate alliance stability over time
        
        Args:
            alliance_strength: Initial alliance strength (0-1 scale)
            internal_cohesion: Member state cohesion factor (0-1)
            external_pressure: External pressure factor (0-1)
            time_periods: Number of time periods
            iterations: Number of Monte Carlo iterations
            crisis_probability: Probability of alliance crisis per period
        
        Returns:
            SimulationResult for alliance stability
        """
        logger.info(f"Running alliance stability simulation: {iterations} iterations")
        
        final_strengths = []
        collapse_count = 0
        
        for _ in range(iterations):
            strength = alliance_strength
            cohesion = internal_cohesion
            
            for period in range(time_periods):
                # Natural decay/strengthening
                drift = np.random.normal(0, 0.05)
                
                # Cohesion affects stability
                cohesion_effect = cohesion * 0.02
                
                # External pressure weakens alliance
                pressure_effect = external_pressure * np.random.uniform(0.01, 0.03)
                
                # Crisis events
                if np.random.random() < crisis_probability:
                    crisis_impact = np.random.uniform(0.05, 0.15)
                    strength -= crisis_impact
                    cohesion -= np.random.uniform(0.02, 0.08)
                
                # Update values
                strength = strength + drift + cohesion_effect - pressure_effect
                strength = max(0, min(1, strength))
                cohesion = max(0, min(1, cohesion))
                
                # Check for collapse
                if strength < 0.2:
                    break
            
            final_strengths.append(strength)
            if strength < 0.3:
                collapse_count += 1
        
        final_strengths = np.array(final_strengths)
        
        result = SimulationResult(
            scenario_type=ScenarioType.ALLIANCE_STABILITY.value,
            iterations=iterations,
            mean=np.mean(final_strengths),
            median=np.median(final_strengths),
            std_dev=np.std(final_strengths),
            percentile_5=np.percentile(final_strengths, 5),
            percentile_25=np.percentile(final_strengths, 25),
            percentile_75=np.percentile(final_strengths, 75),
            percentile_95=np.percentile(final_strengths, 95),
            confidence_interval_95=(
                np.percentile(final_strengths, 2.5),
                np.percentile(final_strengths, 97.5)
            ),
            probability_thresholds={
                "collapse_risk": collapse_count / iterations,
                "weak_alliance_below_0.4": np.mean(final_strengths < 0.4),
                "stable_alliance_above_0.6": np.mean(final_strengths > 0.6)
            },
            metadata={
                "initial_alliance_strength": alliance_strength,
                "internal_cohesion": internal_cohesion,
                "external_pressure": external_pressure,
                "time_periods": time_periods
            }
        )
        
        logger.info(f"Alliance simulation complete. Collapse risk: {result.probability_thresholds['collapse_risk']:.1%}")
        return result
    
    @staticmethod
    def format_result(result: SimulationResult, detailed: bool = True) -> str:
        """
        Format simulation result as human-readable text
        
        Args:
            result: SimulationResult to format
            detailed: Whether to include detailed statistics
        
        Returns:
            Formatted string
        """
        lines = [
            f"Monte Carlo Simulation Results: {result.scenario_type}",
            f"{'=' * 70}",
            f"Iterations: {result.iterations:,}",
            "",
            f"Central Tendency:",
            f"  Mean: {result.mean:.2f}",
            f"  Median: {result.median:.2f}",
            f"  Std Deviation: {result.std_dev:.2f}",
            "",
            f"Confidence Intervals:",
            f"  95% CI: [{result.confidence_interval_95[0]:.2f}, {result.confidence_interval_95[1]:.2f}]",
            f"  5th Percentile: {result.percentile_5:.2f}",
            f"  95th Percentile: {result.percentile_95:.2f}",
        ]
        
        if detailed:
            lines.extend([
                "",
                f"Distribution:",
                f"  25th Percentile: {result.percentile_25:.2f}",
                f"  75th Percentile: {result.percentile_75:.2f}",
                "",
                f"Probability Thresholds:"
            ])
            
            for threshold, prob in result.probability_thresholds.items():
                lines.append(f"  {threshold}: {prob:.1%}")
        
        if result.metadata:
            lines.extend(["", "Input Parameters:"])
            for key, value in result.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


def example_usage():
    """Example usage of Monte Carlo simulator"""
    
    simulator = MonteCarloSimulator(seed=42)
    
    # Example 1: Military equipment attrition
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Russian Tank Stock Projection")
    print("=" * 70)
    
    tank_result = simulator.simulate_military_attrition(
        initial_stock=3000,
        production_rate=200,
        loss_rate=0.15,
        time_periods=5,
        iterations=10000,
        production_uncertainty=0.20,  # High uncertainty due to sanctions
        loss_uncertainty=0.25,
        shock_probability=0.08,  # 8% chance of major battle/loss event per year
        shock_magnitude=0.15
    )
    
    print(MonteCarloSimulator.format_result(tank_result))
    
    # Example 2: Economic sanctions impact
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Economic Sanctions Impact")
    print("=" * 70)
    
    econ_result = simulator.simulate_economic_impact(
        baseline_gdp=2000,  # $2 trillion
        sanctions_impact=0.10,  # 10% initial impact
        adaptation_rate=0.15,  # 15% adaptation per year
        time_periods=5,
        iterations=10000
    )
    
    print(MonteCarloSimulator.format_result(econ_result))
    
    # Example 3: Conflict escalation
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Regional Conflict Escalation Risk")
    print("=" * 70)
    
    conflict_result = simulator.simulate_conflict_escalation(
        initial_intensity=4.0,
        escalation_factors=[0.3, 0.25, 0.2, 0.15, 0.1],
        de_escalation_factors=[0.35, 0.3, 0.2, 0.15],
        time_periods=12,  # months
        iterations=10000
    )
    
    print(MonteCarloSimulator.format_result(conflict_result))


if __name__ == "__main__":
    example_usage()

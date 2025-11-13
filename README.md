# Monte Carlo Geopolitical Simulation Engine  
Probabilistic Modeling for Military, Economic, and Conflict Dynamics

## Overview
This repository contains a from-scratch Monte Carlo simulation engine designed for **geopolitical forecasting, conflict modeling, military attrition analysis, economic shock modeling, and alliance stability assessment**.  
It provides a suite of simulation methods that quantify uncertainty across complex international systems using randomized sampling, probabilistic weighting, and scenario-specific shock events.

All simulations return detailed statistical outputs including:

- Mean, median, standard deviation  
- 5th / 25th / 75th / 95th percentiles  
- 95% confidence intervals  
- Domain-specific probability thresholds  
- Optional multi-period timeline projections  

This module is intended for use in **AI-assisted forecasting pipelines**, **RAG models**, **LLM-based policy analysis**, and **national-security simulations**.

---

## Key Features

### 🔹 **1. Military Attrition Modeling**
Simulates equipment stock over time with:
- Production uncertainty  
- Loss-rate variance  
- Shock events (major battles, sanctions, sudden losses)  
- Full multi-year timelines  

Models support probabilistic estimates such as:
- Probability stock falls below critical levels  
- Probability stock recovers above baseline  
- Risk distributions of depletion  
  
(Implemented in `simulate_military_attrition`.)  
:contentReference[oaicite:1]{index=1}

---

### 🔹 **2. Economic Sanctions Impact Simulation**
Models GDP under sanctions with:
- Adaptive recovery
- Uncertain impact multipliers  
- External shocks (secondary sanctions, supply-chain disruptions)

Outputs contraction probabilities at multiple thresholds:
- <80% baseline  
- <90% baseline  
- <95% baseline  

(Implemented in `simulate_economic_impact`.)  
:contentReference[oaicite:2]{index=2}

---

### 🔹 **3. Conflict Escalation Dynamics**
Simulates conflict intensity (0–10 scale) using:
- Weighted escalation factors  
- Weighted de-escalation factors  
- Diplomatic interventions  
- Major incidents  
- Random-walk drift  

Outputs escalation probabilities, intensity distribution, and risk assessment of >7 intensity.  

(Implemented in `simulate_conflict_escalation`.)  
:contentReference[oaicite:3]{index=3}

---

### 🔹 **4. Alliance Stability Assessment**
Evaluates alliance integrity (0–1 scale) based on:
- Internal cohesion  
- External pressure  
- Crisis probability  
- Natural strengthening/decay  
- Risk of collapse  

Outputs:
- Collapse risk  
- Weak vs. stable alliance probabilities  
- Trajectory distribution  

(Implemented in `simulate_alliance_stability`.)  
:contentReference[oaicite:4]{index=4}

---

### 🔹 **5. Structured Results Format**
All simulations return a `SimulationResult` dataclass containing:
- Central tendency metrics  
- Confidence interval  
- Probability thresholds (scenario-specific)  
- Optional timeline  
- Metadata for reproducibility  

(Defined at top of module.)  
:contentReference[oaicite:5]{index=5}

The module also includes:
- Human-readable result formatter  
- Example usage scripts  
- Optional seed for reproducible simulation  

---

## Repository Structure


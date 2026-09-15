"""Python implementation of the combined-mutation SIS model, where the
pathogen's recovery-evasion trait (Psi_alpha) and transmissibility trait
(Psi_beta) mutate simultaneously -- see phase_diagram_combined.py for the
shared engine and full documentation. In SIS dynamics, recovered nodes are
recycled straight back to susceptible (-1) instead of moving to an
absorbing recovered state, and the outbreak criterion is on the final
infected fraction I/N rather than the recovered fraction R/N. Defaults
match the original alpha/beta models' parameters (N=5000, TT=3000,
alpha=0.1, 50 realizations, full sigma/beta grids, Psi_alpha capped at 20,
Psi_beta capped at 10).
"""
from phase_diagram_combined import main

if __name__ == "__main__":
    main("SIS", default_psi_alpha_cap=20.0, default_psi_beta_cap=10.0,
         prog="PhaseDiagramSIS_combined")

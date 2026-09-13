"""SIRS alpha-mutation phase diagram.

This uses the shared phase_diagram_alpha.py engine. In SIRS dynamics,
infected nodes recover into the -2 compartment, then lose immunity with
probability --omega per step and become susceptible again. The outbreak
criterion is the final infected fraction I/N, matching the endemic-state
logic used by SIS rather than the absorbing-recovered logic used by SIR.
"""
from phase_diagram_alpha import main

if __name__ == "__main__":
    main("SIRS", default_psi_cap=20.0, prog="PhaseDiagramSIRS_alpha")

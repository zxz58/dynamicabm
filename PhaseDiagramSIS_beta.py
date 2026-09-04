"""Python translation of PhaseDiagramSIS_beta.cpp -- see phase_diagram_beta.py
for the shared engine and full documentation. SIS dynamics: recovered nodes
recycle straight back to susceptible, and the outbreak criterion uses the
final infected fraction I/N. Defaults match the original C++ (N=5000,
TT=3000, alpha=0.1, 50 realizations, full sigma/beta grids, Psi capped at
10).
"""
from phase_diagram_beta import main

if __name__ == "__main__":
    main("SIS", default_psi_cap=10.0, prog="PhaseDiagramSIS_beta")

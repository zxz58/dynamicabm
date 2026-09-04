"""Python translation of PhaseDiagramSIS_alpha.cpp -- see phase_diagram_alpha.py
for the shared engine and full documentation. In SIS dynamics, recovered
nodes are recycled straight back to susceptible (-1) instead of moving to
an absorbing recovered state, and the outbreak criterion is on the final
infected fraction I/N rather than the recovered fraction R/N. Defaults
match the original C++ (N=5000, TT=3000, alpha=0.1, 50 realizations, full
sigma/beta grids, Psi capped at 20).
"""
from phase_diagram_alpha import main

if __name__ == "__main__":
    main("SIS", default_psi_cap=20.0, prog="PhaseDiagramSIS_alpha")

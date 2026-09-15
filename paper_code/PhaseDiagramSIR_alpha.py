"""Python translation of PhaseDiagramSIR_alpha.cpp -- see phase_diagram_alpha.py
for the shared engine and full documentation. Run with `python
PhaseDiagramSIR_alpha.py --help` for options; defaults match the original
C++ parameters (N=5000, TT=3000, alpha=0.1, 50 realizations, full sigma/beta
grids, Psi capped at 20).
"""
from phase_diagram_alpha import main

if __name__ == "__main__":
    main("SIR", default_psi_cap=20.0, prog="PhaseDiagramSIR_alpha")

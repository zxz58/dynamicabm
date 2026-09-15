"""Command-line entry point for the community-structured SIRS ABM.

The implementation lives in community_sirs.py so analysis scripts can import
the same engine without duplicating simulation logic.
"""
from community_sirs import main


# Keep this wrapper intentionally small, matching the style of the older
# PhaseDiagram*.py scripts in paper_code/.
if __name__ == "__main__":
    main("CommunitySIRS")

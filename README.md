# Formula checks

Title: Formula checks
Description: Deterministic formula-checking script, recorded outputs, and software requirements for the growth-fragmentation identifiability manuscript.

Files: verify_revision.py, verification_results.json, requirements.txt.
The script, existing output, and requirements were copied unchanged from the
previously supplied growth_fragmentation_revision.zip. They were not regenerated
as part of this formatting-only pass.

Run in this directory:
    python -m pip install -r requirements.txt
    python verify_revision.py --output verification_results.json

The script checks explicit formulas; it is not a PDE simulation or a proof of
long-time convergence. See Appendix A of the manuscript for the precise scope.

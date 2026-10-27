"""Gate experiment entry point: python run_gate_experiment.py [--set all|original|extended]"""
import sys

from src.rag.week9.gate_experiment import main

if __name__ == "__main__":
    sys.exit(main())

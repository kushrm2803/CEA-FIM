from setuptools import setup, find_packages

setup(
    name="cea_fim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.6.0",
        "numpy>=1.19.0", 
        "community>=1.0.0b1",
        "pandas>=1.3.0",
        "numba>=0.53.0"
    ],
    extras_require={
        'gurobi': ['gurobipy>=9.1.0']  # Optional: for LP solver
    }
)
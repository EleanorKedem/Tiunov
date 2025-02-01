# SimCIM TSP Solver

This repository contains an implementation of the SimCIM (Simulated Coherent Ising Machine) algorithm for solving the Traveling Salesman Problem (TSP). The SimCIM approach is a physics-inspired method leveraging continuous-variable systems for efficient optimization. The SimCIM code is based on an algorithm by Tiunove et al. (https://doi.org/10.1364/OE.27.010288), and is adopted with amendments from https://github.com/Human-machine/SimCIM4TSP/tree/main.

## Project Structure

SimCIM.py: Implements the SimCIM algorithm, including amplitude updates, system dynamics, and iterative convergence.
functions.py: Utility functions for handling QUBO matrix operations, energy computations, and parameter conversion.
main.py: The main script that orchestrates the execution of the SimCIM solver on given QUBO instances.
Prerequisites

Ensure you have the following installed:

Python 3.8 or later
Required libraries: NumPy, Torch, Matplotlib
You can install dependencies with:

pip install -r requirements.txt
Usage

Clone this repository:
git clone <repository_url>
cd <repository_directory>
Add your data to the appropriate directory. QUBO problems used for this code can be found in https://github.com/rliang/qubo-benchmark-instances
Run the main script to solve a QUBO instance:
python main.py
Modify parameters in SimCIM.py to adjust optimization settings.
Features

SimCIM Optimization: Implements the Simulated Coherent Ising Machine method for solving QUBO problems.
Amplitude-Based Updates: Uses a dynamic system inspired by physical models.
Modular Structure: The code is organized for easy modification and extension.
Visualization: Includes tools to analyze convergence and solution quality.
Contributions

Feel free to submit issues or pull requests for bug fixes, new features, or improvements.

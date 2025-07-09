## **Solving the 2D Heat Equation using PINNs, FDM, and Analytical Comparison**
This project implements and compares three methods to solve the 2D transient heat equation:

-Physics-Informed Neural Networks (PINNs)

-Finite Difference Method (FDM)

-Analytical Solution (when available)

It provides a visual comparison of accuracy, convergence, and error between the three approaches.

**Problem Statement**: Solving the 2D heat equation:

`∂T/∂t = α ( ∂²T/∂x² + ∂²T/∂y² )`
on a rectangular domain with specified initial and boundary conditions.

**Methods Used**
**PINNs**: Neural networks are trained to satisfy the PDE, initial, and boundary conditions by minimizing a physics-based loss.

**FDM**: Classical numerical technique using finite difference time-stepping and spatial discretization.

**Analytical**: Closed-form solution used for benchmarking when applicable.

**Features**
-Train PINNs with customizable network architectures and learning parameters.

-Numerical solution with FDM for time-evolution.

-Plot and compare results from all methods.

**Requirements**
-Python 3.x

-NumPy

-Matplotlib

-PyTorch

-SciPy

**Output**
-Temperature distribution over time

-Plots comparing all three methods

**License**
Licensed under the MIT License.


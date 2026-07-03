# Parametric Curve Parameter Estimation

## Problem

Estimate theta, M and X from approximately 1500 unordered points generated from a nonlinear parametric curve.

## Approach

### Stage 1
- Generate a dense candidate curve
- Build a KD-Tree
- Compute nearest-neighbor loss
- Optimize using Differential Evolution

### Stage 2
- Recover the continuous t value for every point
- Refine using scipy.optimize.minimize_scalar

### Stage 3
- Refine theta, M and X using nonlinear least squares

Repeat Stage 2 and Stage 3 until convergence.

## Requirements

- numpy
- scipy
- pandas

## Run

```bash
python main.py
```

## Desmos

https://www.desmos.com/calculator/t4mmux4tno


## latex equation


\left(t*\cos(0.523598304142)-e^{0.029999997220\left|t\right|}\cdot\sin(0.3t)\sin(0.523598304142)+54.999998316480,42+t*\sin(0.523598304142)+e^{0.029999997220\left|t\right|}\cdot\sin(0.3t)\cos(0.523598304142)\right)
# Robust regression via PINN

This repository contains the code for the paper:
- [Robust Regression from Highly Corrupted Data via Physics-informed Neural Network](https://arxiv.org/abs/??)

In this work, we propose the Least Absolute Deviation based PINN (LAD-PINN) to reconstruct the solution and recover unknown parameters in PDEs – even if spurious data or outliers corrupt a large percentage of the observations. 
To further improve the accuracy of recovering hidden physics, the two-stage Median Absolute Deviation based PINN (MAD-PINN) is proposed, where LAD-PINN is employed as an outlier detector followed by MAD screening out the highly corrupted data. Then the usual PINN can be applied to the remaining normal data.
Through several examples, including Poisson's equation, wave equation, and steady or unsteady Navier-Stokes equations, we illustrate the power, generality, and efficiency of the proposed algorithms for recovering governing equations from noisy and highly corrupted measurement data.


## Requirements
The requirements are the same with [PINN](https://github.com/maziarraissi/PINNs) and [PINN-laminar-flow](https://github.com/Raocp/PINN-laminar-flow), including:
- Tensorflow 1.x 
- pyDOE
- scipy
- numpy
- pandas
- matplotlib
- silence_tensorflow

## Files
The code is in the form of simple scripts for the [paper](https://arxiv.org/pdf/??). Each script shall be stand-alone and directly runnable.
- `poisson/*` is the Poisson's equation discussed in Section 4.1.
- `piv/*` is the unsteady N-S equation discussed in Section 4.2.
- `wave/*` is the wave equation in Section 4.3.
- `ns/*` is the steady N-S equation in Section 4.4.

## Datasets
- [Steady N-S datasets](https://github.com/Raocp/PINN-laminar-flow/blob/master/PINN_steady/uvNN.pickle)
- [Unsteady N-S datasets](https://github.com/maziarraissi/PINNs/blob/master/main/Data/cylinder_nektar_wake.mat)
The two datasets are also included in the repository.
  
## Citations

```
To be appear.
```

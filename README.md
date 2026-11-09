# BioElec-PINN

JAX implementation of Physics-Informed Neural Network Framework for Performance
Prediction and Parameter Identification of Microbial Fuel Cells

## Notes

- No DAE solver in JAX for the current so we will just need to use a Newton
  solve starting from 0 which should be fine I think.

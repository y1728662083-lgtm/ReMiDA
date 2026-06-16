# Baseline implementation notes

The baseline folder contains the scripts used to compare ReMiDA-CLSRNet with nonlinear ICA-style representation baselines under the fixed-reference real EEG protocol.

## iVAE baseline

The iVAE baseline reuses the iVAE implementation in:

```text
../synthetic_benchmarks/models/nets.py
```

The runner is:

```text
scripts/train_ivae_baseline_paper.py
```

Because vectorized EEG epochs are high-dimensional, the baseline uses standardization and PCA fitted only on `reftrain + targetadapt`, together with VAE stabilization settings such as smaller learning rate, fixed decoder variance, log-variance clipping, gradient clipping, and non-finite-loss checking.

## TCL baseline

The original TensorFlow TCL source is kept in:

```text
../third_party/TCL-master/
```

The submitted baseline uses the PyTorch period-discrimination implementation:

```text
scripts/train_tcl_pytorch_baseline_paper.py
```

This avoids a TensorFlow-1.x dependency while preserving the same period-label discrimination objective used for the baseline comparison.

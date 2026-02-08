# Zigzag Conv MTL12 Layer Configuration Experiment Report

## Experiment Overview
- **Model**: zigzag_conv_mtl12
- **Date**: 2026-02-06
- **Training Data**: UTC 2026-02-01-01
- **Pod**: ai-tf-box-eq-66c57cdfc6-52j45
- **Batch ID**: batch-69a904fb

## Experiments Conducted

### Experiment 1: Compact Architecture
- **Layer Configuration**: [16, 8]
- **Status**: ✅ Completed
- **Duration**: 22 minutes 43 seconds (1362.6s)
- **Description**: Smaller network with 2 hidden layers

### Experiment 2: Deeper Architecture
- **Layer Configuration**: [32, 16, 8]
- **Status**: ✅ Completed
- **Duration**: 15 minutes 21 seconds (921.1s)
- **Description**: Deeper network with 3 hidden layers

## Key Findings

### Training Time Analysis
- The compact architecture [16, 8] took **48% longer** to train (1362.6s vs 921.1s)
- The deeper architecture [32, 16, 8] completed faster despite having more parameters
- This could indicate:
  - Better convergence properties with the deeper architecture
  - Early stopping triggered sooner with more layers
  - Different optimization dynamics

### Configuration Changes
Both experiments modified:
- `ctr_linear_unit_list`: Changed from default [1280, 640, 320, 160, 80, 40, 20] to experimental values
- `cvr_linear_unit_list`: Same modifications applied for consistency
- Total git diff lines: 22 for each experiment

## Recommendations

1. **Architecture Choice**: The 3-layer architecture [32, 16, 8] shows promise with faster training time
2. **Further Investigation**:
   - Check MLflow metrics for accuracy/loss comparisons
   - Analyze model size and inference speed
   - Test intermediate architectures like [32, 16] or [64, 32, 16]
3. **Next Steps**:
   - Compare validation metrics between both architectures
   - Run inference benchmarks
   - Consider hyperparameter tuning for the better performing architecture

## Summary
Both experiments completed successfully. The deeper 3-layer architecture [32, 16, 8] trained significantly faster than the 2-layer [16, 8] configuration, suggesting better training dynamics with the additional layer. Further analysis of model performance metrics is recommended to make the final architecture decision.
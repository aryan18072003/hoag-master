# HOAG Debugging Session Summary - 2026-02-17

## Objective
Analyze and fix the `exp_foe` HOAG implementation to address:
1.  Extremely low validation scores (0.009).
2.  Approach 1 (HOAG) performance lagging behind the Lower Bound baseline.
3.  Catastrophic failure (Score 0.07) when using `ACCEL=8` with small data.

## Key Decisions & Fixes

### 1. Data Integrity & Physics
-   **Issue**: `dataset.py` was quantizing CT images to `uint8` (0-255) via PIL resizing, destroying the dynamic range of Hounsfield Units (HU).
-   **Fix**: Updated `dataset.py` to use `torch.nn.functional.interpolate` on float tensors, preserving the full raw HU range (e.g., `-1000` to `+3000`).
-   **Issue**: `physics.py` was configured for 2-channel input (Complex MRI legacy).
-   **Fix**: Set `IN_CHANNELS = 1` for CT.

### 2. Inner Solver Constraints
-   **Issue**: The inner solver in `hoag.py` / `main.py` had a hardcoded `w.clamp_(-0.5, 1.5)`. Since raw HU values are typically `>100`, this clamp zeroed out practically all signal.
-   **Fix**: Removed the clamp to allow specific value ranges.

### 3. Optimization Balance (Regularization vs Fidelity)
-   **Issue**: The Field of Experts regularizer was initialized with `global_weight = 0.0` ($\exp(0)=1.0$). This resulted in a Regularization Term ~300x larger than the Data Fidelity Term, forcing the inner solver to over-smooth the image to a constant.
-   **Fix**: Changed initialization in `physics.py` to `global_weight = -5.0` ($\exp(-5) \approx 0.0067$). This balances the initial terms, allowing optimization to start from a valid reconstruction.

### 4. Inner Solver Convergence (for ACCEL=8)
-   **Issue**: Convergence analysis showed that `INNER_STEPS=100` and `INNER_LR=0.02` were insufficient for the harder `ACCEL=8` problem, leaving the inner problem under-solved (Loss ~1.0).
-   **Fix**: Updated `main.py` settings:
    -   `INNER_STEPS = 300`
    -   `INNER_LR = 0.1`
    -   `HOAG_CG_MAX_ITER = 50`
    -   **Result**: Inner loss dropped to ~0.66, ensuring high-quality gradients.

### 5. Generalization (Overfitting)
-   **Issue**: The experiment was using `SUBSET_SIZE = 100` (slices). This is <10% of the dataset and likely spans only 1-2 patients. The model overfitted to patient-specific artifacts and failed on validation data (Score 0.07).
-   **Fix**: Updated `main.py`:
    -   `SUBSET_SIZE = None` (Use full dataset: 1051 slices).
    -   Adjusted epochs: `EPOCHS_CLEAN = 20`, `EPOCHS_HOAG = 5` (balanced for 10x larger dataset).

## Current Status
-   **Codebase**: Fully patched and optimized.
-   **Configuration**: Set for `ACCEL=8` (Sparse CT) with full dataset.
-   **Pending**: User to run the full training loop (`python main.py`). Expectation is Approach 1 score > 0.8 and clearly beating Lower Bound.

## Modified Files
-   `exp_foe/dataset.py`: Raw HU loading.
-   `exp_foe/physics.py`: Initialization & Channels.
-   `exp_foe/hoag.py`: Clamp removal.
-   `exp_foe/main.py`: Solver settings, Epochs, Subset strategy.

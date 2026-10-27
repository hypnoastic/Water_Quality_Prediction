# Learning notes

## 1. Define the target before choosing a model

A water-quality index summarizes measurements against chosen objectives. Different
standards answer different questions, so their labels cannot be mixed casually.
Here we predict only the dataset authors' continuous `CCME_Values`.

The audit changes the interpretation: the target is calculated from the same eight
inputs. Excellent scores show **formula emulation**, not independent environmental
knowledge. A time split reduces temporal contamination but cannot remove this
structural dependence. Always check how a label was created.

## 2. Why gradient boosting is a strong starting point

A decision tree splits a table using rules such as “orthophosphate > threshold.”
XGBoost adds trees that correct earlier errors. Shallow trees, shrinkage and
regularization control complexity. It handles threshold-shaped relationships well
and needs no input scaling. A random forest instead averages independently grown
trees to reduce variance; boosting improves a model sequentially.

**Difference here:** the source index contains explicit threshold tests, which suit
trees. The MLP should not be assumed to win simply because it is deep learning.

## 3. What the MLP adds

Each layer learns weighted combinations of its inputs; ReLU adds nonlinearity.
Several layers can learn smooth interactions without manually specifying every
combination. We use 64→32 or 128→64 hidden units and a single regression output.
Dropout, AdamW weight decay and validation early stopping limit overfitting.

`asinh(x)` compresses large values while preserving zero and negative temperatures.
Standardization keeps inputs on comparable scales, making gradient optimization
easier. Both transformations are applied consistently at inference.

**Difference from an LSTM:** a chemistry row is not a time sequence. Treating feature
columns as time steps gives an arbitrary order scientific meaning. An MLP matches
this tabular problem; temporal models need actual sequences and a forecast target.

## 4. A fair comparison is part of the model

Fit imputation and scaling only on historical training rows. Select architecture,
stopping point and blend weight on validation. Use a separate period for interval
calibration, and evaluate the frozen choices on the future test period.
Every model gets the same training sample. Baselines reveal whether extra
complexity buys useful accuracy. A blend helps only if validation supports it.

## 5. Read several kinds of evidence

- **MAE:** average absolute mistake, in index points; easy to interpret.
- **RMSE:** weights large mistakes more heavily; useful for detecting bad misses.
- **R²:** compares squared error with predicting the test mean; can be negative.
- **Permutation importance:** how much validation error increases after shuffling
  a measurement. Correlated inputs can hide each other's importance.
- **TreeSHAP:** allocates a tree prediction among its inputs. It explains the model,
  not the causes of pollution.
- **Residual intervals:** add a held-out residual quantile around a prediction.
  A nominal 90% interval needs observed coverage checks under temporal shift;
  neither trees nor MLPs automatically provide calibrated uncertainty.

## 6. The most useful lesson

Better data and a valid evaluation often matter more than a more complex network.
Preserve extreme pollution events, inspect questionable units, and report missingness
that preprocessing may have hidden. When the exact label formula is available with
all its inputs, compute it directly. To study real predictive value, obtain independent
outcomes, withheld measurements or genuinely future targets.

Further reading: [Breiman, Random Forests (2001)](https://doi.org/10.1023/A:1010933404324),
[Chen & Guestrin, XGBoost (2016)](https://doi.org/10.1145/2939672.2939785),
[Goodfellow et al., chapter 6](https://www.deeplearningbook.org/contents/mlp.html),
[CCME manual (2017)](https://ccme.ca/en/res/wqimanualen.pdf),
[Angelopoulos & Bates, conformal prediction](https://arxiv.org/abs/2107.07511).

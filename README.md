# White Wine Quality: A Logistic Regression Model

Team project for Math 456 (Mathematical Modeling). We use logistic regression
to classify white wines as "good" (quality score >= 7) or not, based on
physicochemical measurements from the UCI Machine Learning Repository's
[Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
(white wine subset, n = 4,898).

**Authors:** Shreyes Balaji, Marwan Hegab, Mazin Hussein, Mikael Rotberg,
Roberto Rubio, Jordan Woda

## Summary

- Binary response `good` = 1 if quality >= 7 (21.6% of wines), 0 otherwise (78.4%).
- 70/30 train/test split (3,428 / 1,470 observations), `set.seed(456)` for reproducibility.
- 208 influential training points (Cook's distance > 4/n) removed before model selection.
- Compared a full 11-predictor model against stepwise-AIC (9 predictors) and
  stepwise-BIC (8 predictors) models; likelihood-ratio tests showed the reduced
  models were not significantly worse than the full model, so the more
  parsimonious BIC model was selected as final.
- Final model: `good ~ chlorides + pH + volatile_acidity + sulphates + free_sulfur_dioxide + residual_sugar + density + fixed_acidity`
- Training fit: ROC AUC = 0.902, McFadden R² = 0.382, Nagelkerke R² = 0.494.
  VIF values are all low except residual sugar (5.16) and density (6.30).
- Classification threshold chosen via Youden's method on the training ROC curve: 0.1485.
- Test-set performance: accuracy = 0.742, precision = 0.438, recall = 0.721,
  specificity = 0.748, F1 = 0.545, ROC AUC = 0.787, log loss = 0.470.
- Example: for a wine at the mean values of the 8 retained predictors, the
  predicted probability of being "good" is 0.058 (95% interval [0.047, 0.070]) —
  below the classification threshold, so it's predicted not good.

Higher pH, sulphates, free sulfur dioxide, residual sugar, and fixed acidity
are associated with higher odds of being a good wine; higher chlorides,
volatile acidity, and density are associated with lower odds. The model
favors recall over precision at this threshold — it catches most good wines
but also flags a fair number of false positives. See the essay for the full
Hosmer–Lemeshow calibration check, ROC curves, and discussion of limitations.

## Files

- [`project_3_essay.pdf`](project_3_essay.pdf) — full write-up (introduction, data
  description, model selection, diagnostics, evaluation, conclusion, references).
- [`analysis.R`](analysis.R) — R code for the full analysis, from data loading
  through stepwise model selection, diagnostics, ROC-based thresholding, and
  test-set evaluation.
- [`data/winequality-white.csv`](data/winequality-white.csv) — the dataset (semicolon-delimited).

## Reproducing the analysis

```r
install.packages(c("tidyverse", "ggpubr", "broom", "car", "MASS", "pROC", "pscl"))
```

Open this folder as your working directory (e.g. open `project_3/` in RStudio),
then run `analysis.R` top to bottom. It reads `data/winequality-white.csv` using
a relative path, so no `setwd()` edits should be needed if the folder structure
is kept intact.

## References

1. UCI Machine Learning Repository — Wine Quality dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
2. FU Berlin SOGA — Logistic Regression in R: https://www.geo.fu-berlin.de/en/v/soga/Basics-of-statistics/Logistic-Regression/Logistic-Regression-in-R---An-Example/index.html
3. STHDA — Logistic Regression Essentials in R: http://www.sthda.com/english/articles/36-classification-methods-essentials/151-logistic-regression-essentials-in-r/

# White Wine Quality: A Logistic Regression Model
#
# How to run:
#   1. Install packages once:
#      install.packages(c("tidyverse","ggpubr","broom","car","MASS","pROC","pscl"))
#   2. Open this project folder in RStudio (or set it as your working directory
#      with setwd()) so the relative path "data/winequality-white.csv" resolves.
#   3. Run the script top to bottom (source it, or run section by section).
#
# Data: UCI Machine Learning Repository, Wine Quality (white wine subset)
#   https://archive.ics.uci.edu/dataset/186/wine+quality
#   Included at data/winequality-white.csv (semicolon-delimited).

# SECTION 0: setup ------------------------------------------------------
library(tidyverse)
library(ggpubr)
library(broom)
library(car)
library(MASS)
library(pROC)
library(pscl)
theme_set(theme_pubr())

# SECTION 1: load data and create binary response ------------------------
wine_raw <- readr::read_delim(
  "data/winequality-white.csv",
  delim = ";",
  show_col_types = FALSE
)

wine <- wine_raw %>%
  rename(
    fixed_acidity = `fixed acidity`,
    volatile_acidity = `volatile acidity`,
    citric_acid = `citric acid`,
    residual_sugar = `residual sugar`,
    free_sulfur_dioxide = `free sulfur dioxide`,
    total_sulfur_dioxide = `total sulfur dioxide`
  ) %>%
  mutate(
    good = if_else(quality >= 7, 1, 0),
    good = as.integer(good)
  )

dim(wine)
names(wine)
summary(wine)
sum(is.na(wine))
colSums(is.na(wine))
table(wine$quality)
prop.table(table(wine$good))

# SECTION 2: exploratory plots ---------------------------------------------
ggplot(wine, aes(x = factor(quality))) +
  geom_bar() +
  labs(
    title = "Distribution of White Wine Quality Scores",
    x = "Quality score",
    y = "Count")

ggplot(wine, aes(x = factor(good))) +
  geom_bar() +
  labs(
    title = "Distribution of Binary Response",
    x = "good (0 = quality < 7, 1 = quality >= 7)",
    y = "Count")

ggplot(wine, aes(x = alcohol)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Distribution of Alcohol",
    x = "Alcohol (%)",
    y = "Count")

# SECTION 3: clean data and train/test split -------------------------------
wine_clean <- wine %>% drop_na()
nrow(wine)
nrow(wine_clean)

set.seed(456)
n <- nrow(wine_clean)
train_idx <- sample(seq_len(n), size = floor(0.70 * n))
train <- wine_clean[train_idx, ]
test  <- wine_clean[-train_idx, ]
nrow(train)
nrow(test)
prop.table(table(train$good))
prop.table(table(test$good))

# SECTION 4: full logistic model -------------------------------------------
predictor_names <- c(
  "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
  "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
  "density", "pH", "sulphates", "alcohol"
)
full_formula <- as.formula(
  paste("good ~", paste(predictor_names, collapse = " + "))
)
null_formula <- good ~ 1

glm_full_train <- glm(
  full_formula,
  data = train,
  family = binomial(link = "logit")
)
glm_null_train <- glm(
  null_formula,
  data = train,
  family = binomial(link = "logit")
)
summary(glm_full_train)

# SECTION 5: remove influential points via Cook's distance -----------------
cook_vals <- cooks.distance(glm_full_train)
cook_cutoff <- 4 / nrow(train)
cook_cutoff
sum(cook_vals > cook_cutoff)

influential_idx <- which(cook_vals > cook_cutoff)
n_influential <- length(influential_idx)
n_influential
train_clean <- train[-influential_idx, ]

nrow(train)
nrow(train_clean)

glm_full_clean <- glm(
  full_formula,
  data = train_clean,
  family = binomial(link = "logit"))

glm_null_clean <- glm(
  null_formula,
  data = train_clean,
  family = binomial(link = "logit"))

summary(glm_full_clean)

# SECTION 6: stepwise selection (AIC and BIC) -------------------------------
glm_step_aic <- step(
  glm_null_clean,
  scope = list(lower = null_formula, upper = full_formula),
  direction = "both",
  trace = 0)

glm_step_bic <- step(
  glm_null_clean,
  scope = list(lower = null_formula, upper = full_formula),
  direction = "both",
  k = log(nrow(train_clean)),
  trace = 0)

summary(glm_step_aic)
summary(glm_step_bic)

attr(terms(glm_step_aic), "term.labels")
attr(terms(glm_step_bic), "term.labels")

# SECTION 7: compare candidate models ---------------------------------------
get_train_metrics <- function(model) {
  tibble(
    AIC = AIC(model),
    BIC = BIC(model),
    LogLik = as.numeric(logLik(model)),
    Deviance = deviance(model),
    McFadden_R2 = pscl::pR2(model)["McFadden"],
    Nagelkerke_R2 = pscl::pR2(model)["r2CU"]
  )}

candidate_models <- list(
  full = glm_full_clean,
  step_aic = glm_step_aic,
  step_bic = glm_step_bic
)

train_metrics <- bind_rows(
  lapply(names(candidate_models), function(nm) {
    get_train_metrics(candidate_models[[nm]]) %>%
      mutate(model = nm, .before = 1)
  }))
train_metrics

anova(glm_step_bic, glm_full_clean, test = "Chisq")
anova(glm_step_aic, glm_full_clean, test = "Chisq")

# SECTION 8: select final model ----------------------------------------------
glm_final <- glm_step_bic
summary(glm_final)
formula(glm_final)

# SECTION 9: multicollinearity check -----------------------------------------
car::vif(glm_final)

# SECTION 10: diagnostics -----------------------------------------------------
diag_df <- tibble(
  fitted = fitted(glm_final),
  deviance_resid = residuals(glm_final, type = "deviance"),
  pearson_resid = residuals(glm_final, type = "pearson"))

ggplot(diag_df, aes(x = fitted, y = deviance_resid)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(
    title = "Deviance Residuals vs Fitted Probabilities",
    x = "Fitted probability",
    y = "Deviance residual")

qqnorm(diag_df$deviance_resid,
       main = "QQ Plot of Deviance Residuals")
qqline(diag_df$deviance_resid, col = 2, lwd = 2)

par(mfrow = c(2, 2))
plot(glm_final)
par(mfrow = c(1, 1))

# SECTION 11: Hosmer-Lemeshow goodness-of-fit test ---------------------------
hoslem_manual <- function(observed, predicted, g = 10) {
  breaks <- quantile(predicted, probs = seq(0, 1, length.out = g + 1))
  groups <- cut(predicted,
                breaks = breaks,
                include.lowest = TRUE)
  tab <- data.frame(
    group = groups,
    obs = observed,
    pred = predicted)
  summary_tab <- tab %>%
    dplyr::group_by(group) %>%
    dplyr::summarise(
      obs_1 = sum(obs),
      obs_0 = n() - sum(obs),
      exp_1 = sum(pred),
      exp_0 = n() - sum(pred),
      n = n(),
      .groups = "drop")
  hl_stat <- sum(
    (summary_tab$obs_1 - summary_tab$exp_1)^2 /
      (summary_tab$exp_1 + 1e-10) +
      (summary_tab$obs_0 - summary_tab$exp_0)^2 /
      (summary_tab$exp_0 + 1e-10)
  )
  df <- g - 2
  p_value <- 1 - pchisq(hl_stat, df)
  list(
    statistic = hl_stat,
    df = df,
    p_value = p_value,
    table = summary_tab)}

hoslem_result <- hoslem_manual(
  observed = train_clean$good,
  predicted = fitted(glm_final),
  g = 10
)
hoslem_result$statistic
hoslem_result$df
hoslem_result$p_value

# SECTION 12: ROC curve and classification threshold (training) -------------
roc_train <- roc(train_clean$good, fitted(glm_final))
auc(roc_train)

plot(roc_train, main = "Training ROC Curve")

best_threshold <- coords(
  roc_train,
  x = "best",
  best.method = "youden",
  ret = "threshold")
best_threshold <- as.numeric(best_threshold)
best_threshold

# SECTION 13: test-set classification metrics --------------------------------
get_classification_metrics <- function(actual, prob, threshold = 0.5) {
  pred <- ifelse(prob >= threshold, 1, 0)

  tp <- sum(actual == 1 & pred == 1)
  tn <- sum(actual == 0 & pred == 0)
  fp <- sum(actual == 0 & pred == 1)
  fn <- sum(actual == 1 & pred == 0)

  accuracy <- (tp + tn) / length(actual)
  precision <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
  specificity <- ifelse(tn + fp == 0, NA, tn / (tn + fp))
  f1 <- ifelse(
    is.na(precision) | is.na(recall) | (precision + recall == 0),
    NA,
    2 * precision * recall / (precision + recall)
  )
  auc_val <- as.numeric(auc(roc(actual, prob)))
  eps <- 1e-15
  prob2 <- pmin(pmax(prob, eps), 1 - eps)
  logloss <- -mean(actual * log(prob2) + (1 - actual) * log(1 - prob2))

  tibble(
    threshold = threshold,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    specificity = specificity,
    f1 = f1,
    auc = auc_val,
    logloss = logloss,
    TP = tp,
    TN = tn,
    FP = fp,
    FN = fn
  )
}

get_test_results <- function(model, model_name, test_data, threshold) {
  prob <- predict(model, newdata = test_data, type = "response")
  get_classification_metrics(test_data$good, prob, threshold) %>%
    mutate(model = model_name, .before = 1)
}

test_results <- bind_rows(
  get_test_results(glm_full_clean, "full", test, best_threshold),
  get_test_results(glm_step_aic, "step_aic", test, best_threshold),
  get_test_results(glm_step_bic, "step_bic", test, best_threshold)
)
test_results

test$prob_final <- predict(glm_final, newdata = test, type = "response")
test$class_final <- ifelse(test$prob_final >= best_threshold, 1, 0)

head(test %>% dplyr::select(good, prob_final, class_final))
table(
  Actual = test$good,
  Predicted = test$class_final)

roc_test_final <- roc(test$good, test$prob_final)
auc(roc_test_final)

plot(roc_test_final, main = "Test ROC Curve - Final Logistic Model")

# SECTION 14: odds ratios -----------------------------------------------------
coef_table <- broom::tidy(glm_final, conf.int = TRUE, exponentiate = TRUE)
coef_table

# SECTION 15: example prediction for an average-profile wine ----------------
final_terms <- attr(terms(glm_final), "term.labels")

new_wine <- train_clean %>%
  summarise(across(all_of(final_terms), mean))
new_wine

pred_link <- predict(glm_final, newdata = new_wine, type = "link", se.fit = TRUE)

eta <- pred_link$fit[1]
se_eta <- pred_link$se.fit[1]

link_lo <- eta - 1.96 * se_eta
link_hi <- eta + 1.96 * se_eta

prob_fit <- plogis(eta)
prob_lo  <- plogis(link_lo)
prob_hi  <- plogis(link_hi)

pred_summary <- tibble(
  logit_fit = eta,
  prob_fit = prob_fit,
  prob_low_95 = prob_lo,
  prob_high_95 = prob_hi)

pred_summary

predicted_class_mean_profile <- ifelse(prob_fit >= best_threshold, 1, 0)
predicted_class_mean_profile

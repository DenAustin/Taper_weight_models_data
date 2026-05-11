library(tidyverse)
library(glmmTMB)
library(DHARMa)
library(performance)
library(splines)
library(readr)
library(ranger)
library(pdp)
library(vip)
library(patchwork)
library(dplyr)
# Set working directory & read data
#Your data source. #e.g. setwd("C:/Users/PC/OneDrive/Desktop/Docum
df_long <- read_csv("df_long_log_1_7B.csv", na = c("", "NA"))

###############################
### SECTION 2: DATA CLEANING AND TRANSFORMATION
###############################

df_long <- df_long %>%
  drop_na() %>%                     # remove rows with any NA
  filter(log_weight >= 0) %>%       # remove impossible negative weights
  mutate(
    # Response transformation
    log_weight_sqrt = sqrt(log_weight),
    
    # Optional sqrt transforms
    DSoB_sqrt = sqrt(DSoB),
    DSuB_sqrt = sqrt(DSuB),
    
    # Convert to factors
    Site        = factor(Site),
    DBH_class   = factor(DBH_class),
    Case        = factor(Case),
    DSoB_class  = factor(DSoB_class),
    DSuB_class  = factor(DSuB_class),
    log_segment = factor(log_segment),
    
    
    # Centered and scaled predictors
    DSoB_c      = DSoB - mean(DSoB, na.rm = TRUE),
    DSoB_s      = scale(DSoB)[, 1],
    DSoB_scaled = scale(DSoB)[, 1],   # alias for clarity
    
    DSuB_c      = DSuB - mean(DSuB, na.rm = TRUE),
    DSuB_s      = scale(DSuB)[, 1],
    DSuB_scaled = scale(DSuB)[, 1],
    
    DBH_c       = DBH - mean(DBH, na.rm = TRUE),
    DBH_s       = scale(DBH)[, 1],
    
    Length_c    = Length - mean(Length, na.rm = TRUE),
    Length_s    = scale(Length)[, 1]
  )

head(df_long)
###############################
### SECTION 2: RANDOM FOREST EDA
###############################



message("Running Random Forest EDA...")

# Fit random forest model
rf_mdl <- ranger(
  formula = log_weight_sqrt ~ DSoB + DSuB + DBH + Length +
    log_segment + Site + DBH_class + DSoB_class + DSuB_class,
  data = df_long,
  importance = "impurity",
  num.trees = 1000,
  respect.unordered.factors = "order"
)

# -----------------------------
# 2.1 Variable Importance
# -----------------------------
var_imp <- rf_mdl$variable.importance
print(var_imp)

# Plot variable importance
vip::vip(rf_mdl, num_features = 10) +
  ggplot2::theme_bw() +
  ggplot2::ggtitle("Random Forest Variable Importance")

# -----------------------------
# 2.2 Partial Dependence Plots (PDP)
# -----------------------------

# PDP for DSoB
pdp_DSoB <- pdp::partial(
  object = rf_mdl,
  pred.var = "DSoB",
  grid.resolution = 50
)
plotPartial(pdp_DSoB)

# PDP for DSuB
pdp_DSuB <- pdp::partial(
  object = rf_mdl,
  pred.var = "DSuB",
  grid.resolution = 50
)
plotPartial(pdp_DSuB)

# PDP for log_segment (optional)
pdp_seg <- pdp::partial(
  object = rf_mdl,
  pred.var = "log_segment"
)
plotPartial(pdp_seg)

# PDP for Site (optional)
pdp_site <- pdp::partial(
  object = rf_mdl,
  pred.var = "Site"
)
plotPartial(pdp_site)

message("Random Forest EDA complete. Review variable importance and PDPs before proceeding to functional-form diagnostics.")



# Convert all plots to ggplot objects
library(pdp)
library(patchwork)

# Convert PDPs to ggplot objects
p1 <- vip::vip(rf_mdl, num_features = 10) +
  ggplot2::theme_bw() +
  ggplot2::ggtitle("Random Forest Variable Importance")

p2 <- autoplot(pdp_DSoB) + ggtitle("PDP: DSoB")
p3 <- autoplot(pdp_DSuB) + ggtitle("PDP: DSuB")
p4 <- autoplot(pdp_seg) + ggtitle("PDP: Segment")

# Combine into 2x2 grid with panel labels A–D
((p1 | p2) /
    (p3 | p4)) +
  patchwork::plot_annotation(tag_levels = "A") # NUMBER POLTS


###############################
### SECTION 3: FUNCTIONAL-FORM DIAGNOSTICS
###############################

message("Running linearity diagnostics for log_weight_sqrt ~ DSoB ...")

# 3.1 Scatterplot with LOESS smoother
ggplot(df_long, aes(DSoB, log_weight_sqrt)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE, color = "blue") +
  theme_bw() +
  ggtitle("Scatterplot with LOESS: Checking linearity")

# 3.2 Fit simple linear model
mdl_lin <- lm(log_weight_sqrt ~ DSoB, data = df_long)
summary(mdl_lin)

# 3.3 Residual diagnostics
par(mfrow = c(1, 2))
plot(mdl_lin, which = 1)   # residuals vs fitted (look for curvature)
plot(mdl_lin, which = 2)   # QQ plot (look for normality)
par(mfrow = c(1, 1))

# 3.4 Component-plus-residual (partial residual) plot
car::crPlots(mdl_lin)

# 3.5 Compare linear vs polynomial vs spline
mdl_poly <- lm(log_weight_sqrt ~ poly(DSoB, 2, raw = TRUE), data = df_long)
mdl_spl  <- lm(log_weight_sqrt ~ splines::bs(DSoB, df = 4), data = df_long)

AIC(mdl_lin, mdl_poly, mdl_spl)

# 3.6 Optional: GAM smooth check
gam_mdl <- mgcv::gam(log_weight_sqrt ~ s(DSoB, k = 10), data = df_long)
summary(gam_mdl)
plot(gam_mdl, residuals = TRUE, pch = 16)

message("Diagnostics complete. Inspect plots and AIC values to determine if linearity holds.")


###############################
### SECTION 4: POLYNOMIAL MODELS (DSoB and DSuB)
###############################

message("Fitting polynomial models using DSoB and DSuB as primary predictors...")

###############################################
### 4A. Polynomial model using DSoB
###############################################

poly_DSoB <- lm(
  log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site,
  data = df_long
)

summary(poly_DSoB)

check_model(poly_DSoB)
# Residual diagnostics
par(mfrow = c(2, 2))
plot(poly_DSoB)
par(mfrow = c(1, 1))

# Additional performance checks (quick)
message("Running performance checks (collinearity, normality, heteroscedasticity)...")
print(check_collinearity(poly_DSoB))       # VIFs for fixed effects
print(check_normality(poly_DSoB))          # normality of residuals
print(check_heteroscedasticity(poly_DSoB)) # heteroscedasticity
##

### ANALYTIC DIAGNOSTICS MODEL 1 DSoB

### Goldfeld-Quandt test

lmtest::gqtest(poly_DSoB, order.by = df_long$DSuB)


ks.test(residuals(poly_DSoB), "pnorm", mean = 0, sd = sd(residuals(poly_DSuB)))

shapiro.test(residuals(poly_DSoB))

lmtest::dwtest(poly_DSoB) ##  Durbin–Watson test

# 
# lmtest::bptest(poly_DSoB)


# lmtest::gqtest(poly_DSoB, order.by = df_long$DSoB)
# 
# ks.test(residuals(poly_DSoB), "pnorm", mean = 0, sd = sd(residuals(poly_DSoB)))
# 
# shapiro.test(residuals(poly_DSoB))

lmtest::dwtest(poly_DSoB) ##  Durbin–Watson test

# DHARMa diagnostics
sim_DSoB <- DHARMa::simulateResiduals(poly_DSoB, n = 1000)
plot(sim_DSoB)

# Partial residual plots
car::crPlots(poly_DSoB)

# Optional GAM comparison
# gam_DSoB <- mgcv::gam(log_weight_sqrt ~ s(DSoB, k = 10) + log_segment + Site,
#                       data = df_long)
# summary(gam_DSoB)
# sim_DSoB <- DHARMa::simulateResiduals(gam_DSoB, n = 1000)
# plot(sim_DSoB)
###############################################
### 4B. Polynomial model using DSuB
###############################################

poly_DSuB <- lm(
  log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site,
  data = df_long
)

summary(poly_DSuB)


# Residual diagnostics

check_model(poly_DSuB)

par(mfrow = c(2, 2))
plot(poly_DSuB)
par(mfrow = c(1, 1))


# 
# lmtest::bptest(poly_DSuB)
### Goldfeld-Quandt test


lmtest::gqtest(poly_DSoB, order.by = df_long$DSuB)

ks.test(residuals(poly_DSuB), "pnorm", mean = 0, sd = sd(residuals(poly_DSoB)))

shapiro.test(residuals(poly_DSuB))

lmtest::dwtest(poly_DSuB) ##  Durbin–Watson test


### Compare Model 1 to Model 2
AIC(poly_DSoB, poly_DSuB)
BIC(poly_DSoB, poly_DSuB)






###############################
### 4C. Train/Test Evaluation for Polynomial Models
###############################

message("Evaluating predictive accuracy for polynomial models (DSoB and DSuB)...")

set.seed(123)

# 80:20 split
n <- nrow(df_long)
train_index <- sample(seq_len(n), size = 0.8 * n)

train_data <- df_long[train_index, ]
test_data  <- df_long[-train_index, ]

###############################################
### Refit polynomial models on training data
###############################################

poly_DSoB_train <- lm(
  log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site,
  data = train_data
)

poly_DSuB_train <- lm(
  log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site,
  data = train_data
)

###############################################
### Predictions on test data
###############################################

pred_DSoB <- predict(poly_DSoB_train, newdata = test_data)
pred_DSuB <- predict(poly_DSuB_train, newdata = test_data)

obs <- test_data$log_weight_sqrt

###############################################
### Remove NA predictions before computing metrics
###############################################

valid_DSoB <- complete.cases(obs, pred_DSoB)
valid_DSuB <- complete.cases(obs, pred_DSuB)

obs_DSoB  <- obs[valid_DSoB]
pred_DSoB <- pred_DSoB[valid_DSoB]

obs_DSuB  <- obs[valid_DSuB]
pred_DSuB <- pred_DSuB[valid_DSuB]

###############################################
### Performance metrics
###############################################

RMSE <- function(obs, pred) sqrt(mean((obs - pred)^2))

R2 <- function(obs, pred) {
  ss_res <- sum((obs - pred)^2)
  ss_tot <- sum((obs - mean(obs))^2)
  1 - ss_res / ss_tot
}

performance_poly <- list(
  Polynomial_DSoB = list(
    RMSE   = RMSE(obs_DSoB, pred_DSoB),
    R2     = R2(obs_DSoB, pred_DSoB),
    AIC    = AIC(poly_DSoB_train),
    BIC    = BIC(poly_DSoB_train),
    logLik = logLik(poly_DSoB_train)
  ),
  Polynomial_DSuB = list(
    RMSE   = RMSE(obs_DSuB, pred_DSuB),
    R2     = R2(obs_DSuB, pred_DSuB),
    AIC    = AIC(poly_DSuB_train),
    BIC    = BIC(poly_DSuB_train),
    logLik = logLik(poly_DSuB_train)
  )
)

performance_poly





###############################
### SECTION 4: POLYNOMIAL MODELS (DSoB and DSuB)
###############################

message("Fitting polynomial models using DSoB and DSuB as primary predictors...")

###############################################
### 4A. Polynomial model using DSoB
###############################################



### ANALYTIC DIAGNOSTICS MODEL 2 DSuB

### BP test for variance constance
lmtest::bptest(poly_DSuB)

### Goldfeld-Quandt test

lmtest::gqtest(poly_DSuB, order.by = df_long$DSuB)


ks.test(residuals(poly_DSuB), "pnorm", mean = 0, sd = sd(residuals(poly_DSuB)))

shapiro.test(residuals(poly_DSuB))

lmtest::dwtest(poly_DSuB) ##  Durbin–Watson test



# DHARMa diagnostics
sim_DSuB <- DHARMa::simulateResiduals(poly_DSuB, n = 1000)
plot(sim_DSuB)

# Partial residual plots
car::crPlots(poly_DSuB)

# Optional GAM comparison
gam_DSuB <- mgcv::gam(log_weight_sqrt ~ s(DSuB, k = 10) + log_segment + Site,
                      data = df_long)
summary(gam_DSuB)

###############################################
### 4C. Compare polynomial models
###############################################

AIC(poly_DSoB, poly_DSuB)
AIC(poly_DSoB, poly_DSuB)
message("Polynomial model diagnostics complete. Review AIC, residuals, and GAM smooths to evaluate suitability.")


###############################################
### SECTION 5: TRAIN/TEST SPLIT + PERFORMANCE
###############################################

set.seed(123)

# 80:20 split
n <- nrow(df_long)
train_index <- sample(seq_len(n), size = 0.8 * n)

train_data <- df_long[train_index, ]
test_data  <- df_long[-train_index, ]

# Refit polynomial models on training data
poly_DSoB_train <- lm(
  log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site,
  data = train_data
)

poly_DSuB_train <- lm(
  log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site,
  data = train_data
)

# Predictions on test data
pred_DSoB <- predict(poly_DSoB_train, newdata = test_data)
pred_DSuB <- predict(poly_DSuB_train, newdata = test_data)

# Compute RMSE
RMSE <- function(obs, pred) sqrt(mean((obs - pred)^2))

rmse_DSoB <- RMSE(test_data$log_weight_sqrt, pred_DSoB)
rmse_DSuB <- RMSE(test_data$log_weight_sqrt, pred_DSuB)

# Compute R² on test data
R2 <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

r2_DSoB <- R2(test_data$log_weight_sqrt, pred_DSoB)
r2_DSuB <- R2(test_data$log_weight_sqrt, pred_DSuB)

# Extract AIC, BIC, logLik from training models
aic_DSoB <- AIC(poly_DSoB_train)
bic_DSoB <- BIC(poly_DSoB_train)
ll_DSoB  <- logLik(poly_DSoB_train)

aic_DSuB <- AIC(poly_DSuB_train)
bic_DSuB <- BIC(poly_DSuB_train)
ll_DSuB  <- logLik(poly_DSuB_train)

# Print results
performance_results <- list(
  DSoB = list(
    RMSE = rmse_DSoB,
    R2   = r2_DSoB,
    AIC  = aic_DSoB,
    BIC  = bic_DSoB,
    logLik = ll_DSoB
  ),
  DSuB = list(
    RMSE = rmse_DSuB,
    R2   = r2_DSuB,
    AIC  = aic_DSuB,
    BIC  = bic_DSuB,
    logLik = ll_DSuB
  )
)

performance_results

### Prediction table
# Create a comparison table for DSoB model
pred_obs_DSoB <- data.frame(
  observed  = test_data$log_weight_sqrt,
  predicted = pred_DSoB,
  log_segment = test_data$log_segment,
  Site = test_data$Site
)

head(pred_obs_DSoB)
tail(pred_obs_DSoB)



poly_DSoB <- lm(
  log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site,
  data = df_long
)

summary(poly_DSoB)

# Residual diagnostics
par(mfrow = c(2, 2))
plot(poly_DSoB)
par(mfrow = c(1, 1))


### ANALYTIC DIAGNOSTICS MODEL 1 DSoB


lmtest::bptest(poly_DSoB)


lmtest::gqtest(poly_DSoB, order.by = df_long$DSoB)

ks.test(residuals(poly_DSoB), "pnorm", mean = 0, sd = sd(residuals(poly_DSoB)))

shapiro.test(residuals(poly_DSoB))

lmtest::dwtest(poly_DSoB) ##  Durbin–Watson test

# DHARMa diagnostics
sim_DSoB <- DHARMa::simulateResiduals(poly_DSoB, n = 1000)
plot(sim_DSoB)

# Partial residual plots
car::crPlots(poly_DSoB)

# Optional GAM comparison
gam_DSoB <- mgcv::gam(log_weight_sqrt ~ s(DSoB, k = 10) + log_segment + Site,
                      data = df_long)
summary(gam_DSoB)

###############################################
### 4B. Polynomial model using DSuB
###############################################

poly_DSuB <- lm(
  log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site,
  data = df_long
)

summary(poly_DSuB)

# Residual diagnostics
par(mfrow = c(2, 2))
plot(poly_DSuB)
par(mfrow = c(1, 1))



### ANALYTIC DIAGNOSTICS MODEL 2 DSuB

### BP test for variance constance
lmtest::bptest(poly_DSuB)

### Goldfeld-Quandt test

lmtest::gqtest(poly_DSuB, order.by = df_long$DSuB)


ks.test(residuals(poly_DSuB), "pnorm", mean = 0, sd = sd(residuals(poly_DSuB)))

shapiro.test(residuals(poly_DSuB))

lmtest::dwtest(poly_DSuB) ##  Durbin–Watson test



# DHARMa diagnostics
sim_DSuB <- DHARMa::simulateResiduals(poly_DSuB, n = 1000)
plot(sim_DSuB)

# Partial residual plots
car::crPlots(poly_DSuB)

# Optional GAM comparison
gam_DSuB <- mgcv::gam(log_weight_sqrt ~ s(DSuB, k = 10) + log_segment + Site,
                      data = df_long)
summary(gam_DSuB)

###############################################
### 4C. Compare polynomial models
###############################################

AIC(poly_DSoB, poly_DSuB)

message("Polynomial model diagnostics complete. Review AIC, residuals, and GAM smooths to evaluate suitability.")


###############################################
### SECTION 5: TRAIN/TEST SPLIT + PERFORMANCE
###############################################

set.seed(123)

# 80:20 split
n <- nrow(df_long)
train_index <- sample(seq_len(n), size = 0.8 * n)

train_data <- df_long[train_index, ]
test_data  <- df_long[-train_index, ]

# Refit polynomial models on training data
poly_DSoB_train <- lm(
  log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site,
  data = train_data
)

poly_DSuB_train <- lm(
  log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site,
  data = train_data
)

# Predictions on test data
pred_DSoB <- predict(poly_DSoB_train, newdata = test_data)
pred_DSuB <- predict(poly_DSuB_train, newdata = test_data)

# Compute RMSE
RMSE <- function(obs, pred) sqrt(mean((obs - pred)^2))

rmse_DSoB <- RMSE(test_data$log_weight_sqrt, pred_DSoB)
rmse_DSuB <- RMSE(test_data$log_weight_sqrt, pred_DSuB)

# Compute R² on test data
R2 <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

r2_DSoB <- R2(test_data$log_weight_sqrt, pred_DSoB)
r2_DSuB <- R2(test_data$log_weight_sqrt, pred_DSuB)

# Extract AIC, BIC, logLik from training models
aic_DSoB <- AIC(poly_DSoB_train)
bic_DSoB <- BIC(poly_DSoB_train)
ll_DSoB  <- logLik(poly_DSoB_train)

aic_DSuB <- AIC(poly_DSuB_train)
bic_DSuB <- BIC(poly_DSuB_train)
ll_DSuB  <- logLik(poly_DSuB_train)

# Print results
performance_results <- list(
  DSoB = list(
    RMSE = rmse_DSoB,
    R2   = r2_DSoB,
    AIC  = aic_DSoB,
    BIC  = bic_DSoB,
    logLik = ll_DSoB
  ),
  DSuB = list(
    RMSE = rmse_DSuB,
    R2   = r2_DSuB,
    AIC  = aic_DSuB,
    BIC  = bic_DSuB,
    logLik = ll_DSuB
  )
)

performance_results

### Prediction table
# Create a comparison table for DSoB model
pred_obs_DSoB <- data.frame(
  observed  = test_data$log_weight_sqrt,
  predicted = pred_DSoB,
  log_segment = test_data$log_segment,
  Site = test_data$Site
)

head(pred_obs_DSoB)
tail(pred_obs_DSoB)



###############################################
### SECTION 5: FIXED vs MIXED POLYNOMIAL MODELS
### TRAIN/TEST SPLIT + PERFORMANCE
###############################################

library(lme4)
library(performance)
library(ggplot2)

## Create Tree unique id as site = cases
df_long$TreeID <- interaction(df_long$Site, df_long$Case, drop = TRUE)



set.seed(123)

# 80:20 split
n <- nrow(df_long)
train_index <- sample(seq_len(n), size = 0.8 * n)

train_data <- df_long[train_index, ]
test_data  <- df_long[-train_index, ]

############################################################
### 5A. FIT FIXED-EFFECTS POLYNOMIAL MODELS (baseline)
############################################################

poly_DSoB_fixed <- lm(
  log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site,
  data = train_data
)

poly_DSuB_fixed <- lm(
  log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site,
  data = train_data
)

############################################################
### 5B. FIT MIXED-EFFECTS POLYNOMIAL MODELS (tree-specific)
############################################################

# Random intercept + random slope for DSoB
poly_DSoB_mixed <- lmer(
  log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site +
    (1 + DSoB | TreeID),
  data = train_data,
  control = lmerControl(optimizer = "bobyqa")
)

# Random intercept + random slope for DSuB
poly_DSuB_mixed <- lmer(
  log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site +
    (1 + DSuB | TreeID),
  data = train_data,
  control = lmerControl(optimizer = "bobyqa")
)

############################################################
### 5C. CHECK CONVERGENCE + RANDOM EFFECTS
############################################################

summary(poly_DSoB_mixed)
summary(poly_DSuB_mixed)

# Convergence diagnostics
check_convergence(poly_DSoB_mixed)
check_convergence(poly_DSuB_mixed)

# Inspect random effects
ranef(poly_DSoB_mixed)
ranef(poly_DSuB_mixed)

############################################################
### 5D. MODEL COMPARISON (AIC, BIC, logLik)
############################################################

model_comparison <- data.frame(
  Model = c("Fixed_DSoB", "Mixed_DSoB", "Fixed_DSuB", "Mixed_DSuB"),
  AIC   = c(AIC(poly_DSoB_fixed), AIC(poly_DSoB_mixed),
            AIC(poly_DSuB_fixed), AIC(poly_DSuB_mixed)),
  BIC   = c(BIC(poly_DSoB_fixed), BIC(poly_DSoB_mixed),
            BIC(poly_DSuB_fixed), BIC(poly_DSuB_mixed)),
  logLik = c(logLik(poly_DSoB_fixed), logLik(poly_DSoB_mixed),
             logLik(poly_DSuB_fixed), logLik(poly_DSuB_mixed))
)

model_comparison

############################################################
### 5E. PREDICTIVE PERFORMANCE ON ORIGINAL SCALE
############################################################

# Back-transform function
back_transform <- function(x) x^2

# Predictions (transformed scale)
pred_fixed_DSoB <- predict(poly_DSoB_fixed, newdata = test_data)
pred_mixed_DSoB <- predict(poly_DSoB_mixed, newdata = test_data, allow.new.levels = TRUE)

pred_fixed_DSuB <- predict(poly_DSuB_fixed, newdata = test_data)
pred_mixed_DSuB <- predict(poly_DSuB_mixed, newdata = test_data, allow.new.levels = TRUE)

# Back-transform
test_data$obs_weight <- back_transform(test_data$log_weight_sqrt)

test_data$pred_fixed_DSoB  <- back_transform(pred_fixed_DSoB)
test_data$pred_mixed_DSoB  <- back_transform(pred_mixed_DSoB)

test_data$pred_fixed_DSuB  <- back_transform(pred_fixed_DSuB)
test_data$pred_mixed_DSuB  <- back_transform(pred_mixed_DSuB)

# RMSE + R2
RMSE <- function(obs, pred) sqrt(mean((obs - pred)^2))
R2   <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

performance <- list(
  DSoB = list(
    RMSE_fixed = RMSE(test_data$obs_weight, test_data$pred_fixed_DSoB),
    RMSE_mixed = RMSE(test_data$obs_weight, test_data$pred_mixed_DSoB),
    R2_fixed   = R2(test_data$obs_weight, test_data$pred_fixed_DSoB),
    R2_mixed   = R2(test_data$obs_weight, test_data$pred_mixed_DSoB)
  ),
  DSuB = list(
    RMSE_fixed = RMSE(test_data$obs_weight, test_data$pred_fixed_DSuB),
    RMSE_mixed = RMSE(test_data$obs_weight, test_data$pred_mixed_DSuB),
    R2_fixed   = R2(test_data$obs_weight, test_data$pred_fixed_DSuB),
    R2_mixed   = R2(test_data$obs_weight, test_data$pred_mixed_DSuB)
  )
)

performance

############################################################
### 5F. CHECK REDUCTION OF UNDER-PREDICTION FOR LOG1
############################################################


bias_by_segment <- test_data %>%
  group_by(log_segment) %>%
  summarise(
    mean_obs = mean(obs_weight),
    fixed_pred = mean(pred_fixed_DSoB),
    mixed_pred = mean(pred_mixed_DSoB),
    fixed_bias = mean(pred_fixed_DSoB - obs_weight),
    mixed_bias = mean(pred_mixed_DSoB - obs_weight)
  )

bias_by_segment

######### DSuB 
bias_by_segment_uB <- test_data %>%
  group_by(log_segment) %>%
  summarise(
    mean_obs = mean(obs_weight),
    fixed_pred = mean(pred_fixed_DSuB),
    mixed_pred = mean(pred_mixed_DSuB),
    fixed_bias = mean(pred_fixed_DSuB - obs_weight),
    mixed_bias = mean(pred_mixed_DSuB - obs_weight)
  )

bias_by_segment_uB

############################################################
### 5G. PREDICTED vs OBSERVED PLOTS
############################################################

ggplot(test_data, aes(x = obs_weight, y = pred_fixed_DSoB)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  ggtitle("Fixed Polynomial (DSoB): Pred. vs Obsd.")

ggplot(test_data, aes(x = obs_weight, y = pred_mixed_DSoB)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  ggtitle("Mixed Polynomial (DSoB): Pred. vs Obsd.")


ggplot(test_data, aes(x = obs_weight, y = pred_fixed_DSuB)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  ggtitle("Fixed Polynomial (DSuB): Pred. vs Obsd.")

ggplot(test_data, aes(x = obs_weight, y = pred_mixed_DSuB)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  ggtitle("Mixed Polynomial (DSuB): Pred. vs Obsd.")


library(patchwork)
p1 <- ggplot(test_data, aes(x = obs_weight, y = pred_fixed_DSoB)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  ggtitle("Fixed Polynomial (DSoB): Pred. vs Obsd.")

p2 <- ggplot(test_data, aes(x = obs_weight, y = pred_mixed_DSoB)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  ggtitle("Mixed Polynomial (DSoB): Pred. vs Obsd.")

p3 <- ggplot(test_data, aes(x = obs_weight, y = pred_fixed_DSuB)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  ggtitle("Fixed Polynomial (DSuB): Pred. vs Obsd.")

p4 <- ggplot(test_data, aes(x = obs_weight, y = pred_mixed_DSuB)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_bw() +
  ggtitle("Mixed Polynomial (DSuB): Pred. vs Obsd.")


# Side-by-side (1×2)
p1 | p2

# Stacked (2×1)
p1 / p2

# 2×2 layout
(p1 | p2) /
(p3 | p4)

###############################################
### SECTION 5: 5-FOLD CV FOR FIXED vs MIXED POLYNOMIAL MODELS
###############################################

# library(lme4)
# library(performance)
# library(dplyr)
# library(ggplot2)
# 
# # Ensure TreeID exists
# df_long$TreeID <- interaction(df_long$Site, df_long$Case, drop = TRUE)
# 
# # Back-transform function
# back_transform <- function(x) x^2
# 
# # Metrics
# RMSE <- function(obs, pred) sqrt(mean((obs - pred)^2))
# R2   <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
# 
# set.seed(123)
# 
# ###############################################
# ### 5A. CREATE 5 FOLDS BY TREEID
# ###############################################
# 
# tree_ids <- unique(df_long$TreeID)
# K <- 5
# fold_assignments <- sample(rep(1:K, length.out = length(tree_ids)))
# names(fold_assignments) <- tree_ids
# 
# df_long$Fold <- fold_assignments[df_long$TreeID]
# 
# ###############################################
# ### 5B. STORAGE FOR RESULTS
# ###############################################
# 
# results_fixed <- list()
# results_mixed <- list()
# bias_results <- list()
# 
# ###############################################
# ### 5C. RUN 5-FOLD CV
# ###############################################
# 
# for (k in 1:K) {
#   
#   cat("\nRunning fold", k, "...\n")
#   
#   train_data <- df_long %>% filter(Fold != k)
#   test_data  <- df_long %>% filter(Fold == k)
#   
#   ### FIXED MODELS
#   poly_DSoB_fixed <- lm(
#     log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site,
#     data = train_data
#   )
#   
#   poly_DSuB_fixed <- lm(
#     log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site,
#     data = train_data
#   )
#   
#   ### MIXED MODELS
#   poly_DSoB_mixed <- lmer(
#     log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site +
#       (1 + DSoB | TreeID),
#     data = train_data,
#     control = lmerControl(optimizer = "bobyqa")
#   )
#   
#   poly_DSuB_mixed <- lmer(
#     log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site +
#       (1 + DSuB | TreeID),
#     data = train_data,
#     control = lmerControl(optimizer = "bobyqa")
#   )
#   
#   ### PREDICT
#   test_data$obs_weight <- back_transform(test_data$log_weight_sqrt)
#   
#   # DSoB
#   test_data$pred_fixed_DSoB <- back_transform(
#     predict(poly_DSoB_fixed, newdata = test_data)
#   )
#   
#   test_data$pred_mixed_DSoB <- back_transform(
#     predict(poly_DSoB_mixed, newdata = test_data, allow.new.levels = TRUE)
#   )
#   
#   # DSuB
#   test_data$pred_fixed_DSuB <- back_transform(
#     predict(poly_DSuB_fixed, newdata = test_data)
#   )
#   
#   test_data$pred_mixed_DSuB <- back_transform(
#     predict(poly_DSuB_mixed, newdata = test_data, allow.new.levels = TRUE)
#   )
#   
#   ### STORE PERFORMANCE
#   results_fixed[[k]] <- c(
#     RMSE_DSoB = RMSE(test_data$obs_weight, test_data$pred_fixed_DSoB),
#     RMSE_DSuB = RMSE(test_data$obs_weight, test_data$pred_fixed_DSuB),
#     R2_DSoB   = R2(test_data$obs_weight, test_data$pred_fixed_DSoB),
#     R2_DSuB   = R2(test_data$obs_weight, test_data$pred_fixed_DSuB)
#   )
#   
#   results_mixed[[k]] <- c(
#     RMSE_DSoB = RMSE(test_data$obs_weight, test_data$pred_mixed_DSoB),
#     RMSE_DSuB = RMSE(test_data$obs_weight, test_data$pred_mixed_DSuB),
#     R2_DSoB   = R2(test_data$obs_weight, test_data$pred_mixed_DSoB),
#     R2_DSuB   = R2(test_data$obs_weight, test_data$pred_mixed_DSuB)
#   )
#   
#   ### SEGMENT-SPECIFIC BIAS
#   bias_results[[k]] <- test_data %>%
#     group_by(log_segment) %>%
#     summarise(
#       mean_obs = mean(obs_weight),
#       fixed_pred = mean(pred_fixed_DSoB),
#       mixed_pred = mean(pred_mixed_DSoB),
#       fixed_bias = mean(pred_fixed_DSoB - obs_weight),
#       mixed_bias = mean(pred_mixed_DSoB - obs_weight)
#     )
# }
# 
# ###############################################
# ### 5D. AGGREGATE RESULTS
# ###############################################
# 
# fixed_summary <- do.call(rbind, results_fixed) %>% as.data.frame()
# mixed_summary <- do.call(rbind, results_mixed) %>% as.data.frame()
# 
# cat("\n\n=== FIXED MODEL CV RESULTS ===\n")
# apply(fixed_summary, 2, mean)
# 
# cat("\n\n=== MIXED MODEL CV RESULTS ===\n")
# apply(mixed_summary, 2, mean)
# 
# ###############################################
# ### 5E. AGGREGATE SEGMENT-SPECIFIC BIAS
# ###############################################
# 
# bias_df <- bind_rows(bias_results, .id = "Fold")
# 
# bias_summary <- bias_df %>%
#   group_by(log_segment) %>%
#   summarise(
#     mean_obs = mean(mean_obs),
#     fixed_bias = mean(fixed_bias),
#     mixed_bias = mean(mixed_bias)
#   )
# 
# bias_summary
# 
# 
# 
# ################## PART II #######################################
# ################               ################################
# ############ GLMMs ######### GLMMs ###################
# 
# 
# ############## MODELS 1 and 2 ##############
# 
# 
# 
# ###########  Model 4   ###

###############################################
### 5-FOLD CROSS-VALIDATION FOR MODELS 1 & 2
### Polynomial LLMs (DSoB and DSuB)
###############################################

library(dplyr)
library(purrr)

set.seed(123)

# Remove rows with NA in required variables
df_cv_poly <- df_long %>%
  drop_na(log_weight_sqrt, DSoB, DSuB, log_segment, Site)

# Create 5 folds
df_cv_poly$fold <- sample(rep(1:5, length.out = nrow(df_cv_poly)))

# RMSE and R2 functions
RMSE <- function(obs, pred) sqrt(mean((obs - pred)^2))
R2 <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

# Storage
cv_poly_results <- list(Model1 = list(), Model2 = list())

for (k in 1:5) {
  
  train_data <- df_cv_poly %>% filter(fold != k)
  test_data  <- df_cv_poly %>% filter(fold == k)
  
  # ---- Model 1: DSoB polynomial ----
  m1 <- lm(
    log_weight_sqrt ~ DSoB + I(DSoB^2) + log_segment + Site,
    data = train_data
  )
  
  pred1 <- predict(m1, newdata = test_data)
  
  valid1 <- complete.cases(pred1, test_data$log_weight_sqrt)
  cv_poly_results$Model1[[k]] <- list(
    RMSE = RMSE(test_data$log_weight_sqrt[valid1], pred1[valid1]),
    R2   = R2(test_data$log_weight_sqrt[valid1], pred1[valid1])
  )
  
  # ---- Model 2: DSuB polynomial ----
  m2 <- lm(
    log_weight_sqrt ~ DSuB + I(DSuB^2) + log_segment + Site,
    data = train_data
  )
  
  pred2 <- predict(m2, newdata = test_data)
  
  valid2 <- complete.cases(pred2, test_data$log_weight_sqrt)
  cv_poly_results$Model2[[k]] <- list(
    RMSE = RMSE(test_data$log_weight_sqrt[valid2], pred2[valid2]),
    R2   = R2(test_data$log_weight_sqrt[valid2], pred2[valid2])
  )
}

# Summaries
cv_poly_summary <- list(
  Model1_DSoB = map_df(cv_poly_results$Model1, ~as.data.frame(.x)) %>% summarise_all(mean),
  Model2_DSuB = map_df(cv_poly_results$Model2, ~as.data.frame(.x)) %>% summarise_all(mean)
)

cv_poly_summary

AIC(m_final, m_final_oB)
BIC(m_final, m_final_oB)


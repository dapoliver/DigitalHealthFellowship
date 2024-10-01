##### Load necessary libraries #####
library(tidyverse)  # For data manipulation and visualization
library(scales)     # To rescale numeric data
library(glmnet)     # For model fitting
library(yardstick)  # For measuring model performance
library(nestedcv)   # For performing nested cross-validation
library(probably)   # For improving model probability thresholds
library(caret)      # For building and tuning machine learning models
library(survival)   # For concordance function
library(MLmetrics) # For F1_Score
library(mice) # For multiple imputation

# Set working directory
setwd("~/Dropbox/Work/Fellowships/")

# Create folders for saving results
dir <- file.path("Data") 
if (!dir.exists(dir)) dir.create(dir)
dir <- file.path("Results", "1. Performance") 
if (!dir.exists(dir)) dir.create(dir)
dir <- file.path("Results", "2. Variable Importance") 
if (!dir.exists(dir)) dir.create(dir)
dir <- file.path("Results", "3. Model Evaluation") 
if (!dir.exists(dir)) dir.create(dir)

##### Create a synthetic dataset with processed features #####
synth <- data.frame(
  # Feature 1-6: ENTER measures
  enter_pq16 = rescale(rbeta(1500, 0.5, 2), to = c(0, 16)),
  enter_bpss = rescale(rbeta(1500, 0.5, 2), to = c(0, 140)),
  enter_pps = rescale(rbeta(1500, 0.5, 2), to = c(0, 37)),
  enter_predictd = as.factor(sample(c("Yes", "No"), 1500, replace = TRUE, prob = c(0.05, 0.95))),
  enter_dst = rescale(rbeta(1500, 0.5, 2), to = c(0, 100)),
  enter_speech = as.factor(sample(c("Yes", "No"), 1500, replace = TRUE, prob = c(0.05, 0.95))),
  
  # Features 7-10: Mean time spent at home, screen, sleep, and active activities derived from mindLAMP
  home_time_mean = rescale(rnorm(1500, mean = 10, sd = 5), to = c(0, 24)),
  screen_time_mean = rescale(rnorm(1500, mean = 6, sd = 1), to = c(0, 16)),
  sleep_mean = rescale(rnorm(1500, mean = 8, sd = 1), to = c(0, 16)),
  active_time_mean = rescale(rnorm(1500, mean = 4, sd = 2), to = c(0, 8)),
  
  # Features 11-12: Mean number of messages and call time (hours) collected by mindLAMP
  messages_mean = rescale(rnorm(1500, mean = 200, sd = 100), to = c(0, 500)),
  calls_mean = rescale(rnorm(1500, mean = 2, sd = 8), to = c(0, 16)),
  
  # Features 13-16: RMSSD (Root Mean Square of Successive Differences) for mindLAMP passive data
  home_time_rmssd = rescale(rnorm(1500, mean = 10, sd = 5), to = c(0, 24)),
  screen_time_rmssd = rescale(rnorm(1500, mean = 6, sd = 1), to = c(0, 16)),
  sleep_rmssd = rescale(rnorm(1500, mean = 8, sd = 1), to = c(0, 16)),
  active_time_rmssd = rescale(rnorm(1500, mean = 4, sd = 2), to = c(0, 8)),
  
  # Feature 17-19: Sociodemographics
  age = rescale(rnorm(1500, mean = 24, sd = 8), to = c(16, 35)),
  gender = as.factor(sample(c("Male", "Female", "Other"), 1500, replace = TRUE, prob = c(0.45, 0.45, 0.1))),
  ethnicity = as.factor(sample(c("White", "Black", "Asian", "Mixed", "Other"), 1500, replace = TRUE, prob = c(0.6, 0.13, 0.13, 0.06, 0.13))),
  
  # Features 20-25: Mean for symptoms, functioning and quality of life measured by mindLAMP
  pq16_mean = rescale(rbeta(1500, 0.5, 2), to = c(0, 16)),
  bpss_mean = rescale(rbeta(1500, 0.5, 2), to = c(0, 140)),
  phq_mean = rescale(rbeta(1500, 0.5, 2), to = c(0, 27)),
  oasis_mean = rescale(rbeta(1500, 0.5, 2), to = c(0, 25)),
  functioning_mean = rescale(rbeta(1500, 0.5, 2), to = c(0, 30)),
  qol_mean = rescale(rbeta(1500, 2, 0.5), to = c(0, 100)),
  
  # Features 26-31: RMSSD for symptoms, functioning and quality of life measured by mindLAMP
  pq16_rmssd = rescale(rbeta(1500, 0.5, 2), to = c(0, 16)),
  bpss_rmssd = rescale(rbeta(1500, 0.5, 2), to = c(0, 140)),
  phq_rmssd = rescale(rbeta(1500, 0.5, 2), to = c(0, 27)),
  oasis_rmssd = rescale(rbeta(1500, 0.5, 2), to = c(0, 25)),
  functioning_rmssd = rescale(rbeta(1500, 0.5, 2), to = c(0, 30)),
  qol_rmssd = rescale(rbeta(1500, 2, 0.5), to = c(0, 100)),
  
  # Features 32-34: Binary indicators for psychiatric medications (antidepressants, anxiolytics, antipsychotics)
  antidepressants = as.factor(sample(c("Yes", "No"), 1500, replace = TRUE, prob = c(0.05, 0.95))),
  anxiolytics = as.factor(sample(c("Yes", "No"), 1500, replace = TRUE, prob = c(0.05, 0.95))),
  antipsychotics = as.factor(sample(c("Yes", "No"), 1500, replace = TRUE, prob = c(0.05, 0.95))),
  
  # Outcome: Chronic serious mental disorder (CHR-SMD), 40% Yes, 60% No
  chrsmd = sample(c("Yes", "No"), 1500, replace = TRUE, prob = c(0.4, 0.6))
)

# Randomly select indices to introduce missing values for the outcome (representing participants who do not complete stage 2)
missing_indices <- sample(1:nrow(synth), 1100)
synth$chrsmd[missing_indices] <- NA

# Randomly select datapoints to introduce missing predictor values
missing_vals <- data.frame(row=sample(1:nrow(synth), 1100,replace=TRUE),
                           col=sample(1:ncol(synth), 1100,replace=TRUE))

for (i in 1:nrow(missing_vals)) {
  row <- missing_vals$row[i]
  col <- missing_vals$col[i]
  synth[row, col] <- NA
}

write_csv(synth,"Data/synthetic_data.csv")

##### Parameters for nested cross-validation #####
outerFolds <- 10        # Number of outer cross-validation folds
outerRepeats <- 10      # Number of repetitions for outer cross-validation
innerFolds <- 10        # Number of inner cross-validation folds
innerRepeats <- 10     # Number of repetitions for inner cross-validation

# Create a function to perform repeated nested cross-validation for random forest
repeated_nested_cv_lr <- function(combined_df, outerFolds, outerRepeats, innerFolds, innerRepeats, tuneGrid, seed) {
  
  set.seed(seed)
  seeds <- sample(1:10000, outerRepeats) # Generate unique seeds for each outer repeat
  
  best_inner_result_list <- list()
  best_lambda_list <- list()
  all_inner_results <- list()
  c_stat_nested <- data.frame()
  coef_df <- data.frame()
  c_stat_nested_ethnicity <- data.frame()
  c_stat_nested_gender <- data.frame()
  # Outer loop for cross-validation
  for (outer_rep in 1:outerRepeats) {
    cat("Outer Repeat:", outer_rep, "\n")
    set.seed(seeds[outer_rep]) # Set a unique seed for each outer repeat
    labelled <- combined_df %>% filter(!is.na(chrsmd))
    unlabelled <- combined_df %>% filter(is.na(chrsmd))
    outer_folds <- createFolds(labelled$chrsmd, k = outerFolds, list = TRUE, returnTrain = TRUE)
    
    for (i in seq_along(outer_folds)) {
      cat("  Outer Fold:", i, "\n")
      labelled <- combined_df %>% filter(!is.na(chrsmd))
      unlabelled <- combined_df %>% filter(is.na(chrsmd))
      train_indices <- outer_folds[[i]]
      train <- labelled[train_indices, ] #  This splits the data into train (inner loop)
      test <- labelled[-train_indices, ] #  This splits the data into test (outer loop)
      
      # Impute test dataset
      mice_imputes_test = mice(test, m = 5, maxit = 5, seed = 12345, print=FALSE)
      test <- complete(mice_imputes_test)
      
      # Impute unlabelled data except for outcome
      mice_imputes_unlabelled = mice(unlabelled[,c(1:9)], m = 5, maxit = 5, seed = 12345, print=FALSE)
      unlabelled[,c(1:9)] <- complete(mice_imputes_unlabelled)
      
      # Inner Cross-Validation and Model Training
      for (ssl in 1:10){     
        cat("    Semi-Supervised Learning Iteration:", ssl, "\n")
        
        inner_results <- vector("list", innerRepeats) # Initialize as a list to store results for each inner repeat
        
        # Impute training data
        mice_imputes_train = mice(train, m = 5, maxit = 5, seed = 12345, print=FALSE)
        train <- complete(mice_imputes_train)
        
        for (inner_rep in 1:innerRepeats) {
          cat("    Inner Repeat:", inner_rep, "\n")
          
          # Use a different seed for each inner repeat
          set.seed(seeds[outer_rep] + inner_rep)
          
            # Define a custom summary function to calculate F1 score
          customSummary <- function(data, lev = NULL, model = NULL) {
            f1 <- F1_Score(y_pred = data$pred, y_true = data$obs, positive = lev[1])
            out <- c(F1 = f1)
            out
          }
            
            # Random grid search to generate 10 lambda values 
            tune_grid_temp <- sample_n(tune_grid_lr,10)
            
            # Set up the trainControl with the custom summary function
            control <- trainControl(method = 'cv', 
                                    number = 5, 
                                    classProbs = TRUE,
                                    summaryFunction = customSummary,
                                    search = 'random')
            
            # Train the model using the custom F1 metric
            inner_model <- train(chrsmd ~ .,
                                 data = train,
                                 method = "glmnet",
                                 metric = "F1",
                                 
                                 tuneGrid = tune_grid_temp,
                                 trControl = control)
            
            print(inner_model)
            
            if (all(!is.na(inner_model$results$F1))){
              # Store the F1 for each inner repeat
              inner_results[[inner_rep]] <- rbind(inner_results[[inner_rep]], 
                                                  data.frame(lambda = inner_model$bestTune$lambda,
                                                             F1 = inner_model$results$F1, 
                                                             repeat_number = inner_rep,
                                                             ssl = ssl))
            }
        }
        
        # Combine results from all repeats
        all_inner_results <- do.call(rbind, inner_results)
        print(dim(all_inner_results))
        
        # Calculate the average F1 for each hyperparameter combination
        avg_results <- all_inner_results %>%
          group_by(lambda) %>%
          summarise(avg_F1 = mean(F1, na.rm=TRUE),
                    lambda = mean(lambda, na.rm=TRUE),
                    .groups = "keep") 
        
        # Determine the best hyperparameter combination based on the highest average F1
        best_inner_result <- avg_results[which.max(avg_results$avg_F1), ]
        best_lambda <- best_inner_result$lambda
        best_F1 <- best_inner_result$avg_F1

        # Calculate the correct index for the current outer fold and repeat
        index <- (outer_rep - 1) * outerFolds + i
        
        # Store the best hyperparameters for the current outer fold
        best_inner_result_list[[index]] <- best_inner_result
        best_lambda_list[[index]]  <- best_lambda

        # Train the final model using the best hyperparameters on the inner training set (x_inner_train, y_inner_train)
        control_final <- trainControl(method = 'none')
        repGrid <- data.frame(alpha=1,
                              lambda=best_lambda)  
        
        ##### Fit best model from inner repeat to generate pseudolabels
        best_inner_model <- train(chrsmd ~ .,
                                  data = train,
                                  method = "glmnet",
                                  metric = "F1",
                                  trControl=control_final,
                                  tuneGrid = repGrid)
        
        # Generate pseudolabels and apply confidence threshold
        unlabelled <- unlabelled %>% mutate(PI = predict(best_inner_model, newdata = unlabelled, type = "prob")[,2],
                                            confident = case_when(PI>0.7 | PI<0.3 ~ 1,
                                                                  TRUE ~ 0),
                                            chrsmd = case_when(PI>0.7 ~ "Yes",
                                                               PI<0.3 ~ "No"))
        
        # Filter datasets so you have new confident_pseudolabels dataset and remove them from unlabelled
        confident_pseudolabels <- unlabelled %>% filter(confident==1)
        confident_pseudolabels <- confident_pseudolabels %>% subset(select=c(-confident, -PI))
        unlabelled <- unlabelled %>% filter(confident==0)
        
        # Stop semi-supervised learning if less than 10 participants have pseudolabels with high confidence 
        if (nrow(confident_pseudolabels)<10) {
          print(paste0('Minimal confident pseudolabels after ', ssl, ' iterations'))
          break
        }
        
        # Merge dataframes
        train <- rbind(train, confident_pseudolabels)
      }
    }
      final_model <- train(chrsmd ~ .,
                           data = train,
                           method = "glmnet",
                           metric = "F1",
                           trControl=control_final,
                           tuneGrid = repGrid)
      
      # Extract coefficients
      best_model <- final_model$finalModel
      best_lambda <- final_model$bestTune$lambda
      coef_matrix <- as.matrix(coef(best_model, s = best_lambda))
      coef_df <- rbind(coef_df, data.frame(Feature = rownames(coef_matrix), Coefficient = coef_matrix[, 1]))
      
      # Predict the linear predictors (PI) from the Elastic Net model
      test$PI <- predict(final_model, newdata = test, type = "prob")[,2]
      test$chrsmd_pred <- predict(final_model, newdata = test, type = "raw")
      test$chrsmd_f <- factor(test$chrsmd)
      
      cm <- confusionMatrix(data = test$chrsmd_pred, reference = test$chrsmd_f)
      test <- test %>% mutate(PI=case_when(PI==0 ~ 0.001, PI==1 ~ 0.999, TRUE ~ PI))
      
      # Fit a Cox model on the test set using penalized coefficients
      model_test <- glm(chrsmd_f ~ PI, data = test, family="binomial")
      
      calibration <- rms::val.prob(p=test$PI, y=as.numeric(test$chrsmd_f)-1, m=200, pl=F)
      
      # Calculate calibration slope on the hold-out data (test set)
      calibration_intercept <- unname(calibration[12])
      calibration_slope <- unname(calibration[13])
      brier <- unname(calibration[11])
      
      c_stat_nested <- rbind(c_stat_nested, data.frame(
        C_test = concordance(model_test)$concordance,
        SE_test = concordance(model_test)$cvar,
        Fold = i,
        OuterRepeat = outer_rep,
        n_train = nrow(train),
        events_train = sum(train$chrsmd=="Yes"),
        n_test = nrow(test),
        events_test = sum(test$chrsmd=="Yes"),
        balanced_accuracy = cm$byClass[11],
        sensitivity = cm$byClass[1],
        specificity = cm$byClass[2],
        ppv = cm$byClass[3],
        npv = cm$byClass[4],
        precision = cm$byClass[5],
        recall = cm$byClass[6],
        f1 = cm$byClass[7],
        intercept = calibration_intercept,
        slope = calibration_slope,
        brier = brier
      ))
      
  # Test algorithmic bias
  white <- test %>% filter(ethnicity=="White")
  black <- test %>% filter(ethnicity=="Black")
  asian <- test %>% filter(ethnicity=="Asian")
  mixed <- test %>% filter(ethnicity=="Mixed")
  other_eth <- test %>% filter(ethnicity=="Other")
  male <- test %>% filter(gender=="Male")
  female <- test %>% filter(gender=="Female")
  other_gen <- test %>% filter(gender=="Other")
  
  m.white <- glm(chrsmd_f ~ PI, data = white, family="binomial")
  m.black <- glm(chrsmd_f ~ PI, data = black, family="binomial")
  m.asian <- glm(chrsmd_f ~ PI, data = asian, family="binomial")
  m.mixed <- glm(chrsmd_f ~ PI, data = mixed, family="binomial")
  m.other_eth <- glm(chrsmd_f ~ PI, data = other_eth, family="binomial")
  m.male <- glm(chrsmd_f ~ PI, data = male, family="binomial")
  m.female <- glm(chrsmd_f ~ PI, data = female, family="binomial")
  m.other_gen <- glm(chrsmd_f ~ PI, data = other_gen, family="binomial")
  
  cal_white <- rms::val.prob(p=white$PI, y=as.numeric(white$chrsmd_f)-1, m=200, pl=F)
  cal_black <- rms::val.prob(p=black$PI, y=as.numeric(black$chrsmd_f)-1, m=200, pl=F)
  cal_asian <- rms::val.prob(p=asian$PI, y=as.numeric(asian$chrsmd_f)-1, m=200, pl=F)
  cal_mixed <- rms::val.prob(p=mixed$PI, y=as.numeric(mixed$chrsmd_f)-1, m=200, pl=F)
  cal_other_eth <- rms::val.prob(p=other_eth$PI, y=as.numeric(other_eth$chrsmd_f)-1, m=200, pl=F)
  cal_male <- rms::val.prob(p=male$PI, y=as.numeric(male$chrsmd_f)-1, m=200, pl=F)
  cal_female <- rms::val.prob(p=female$PI, y=as.numeric(female$chrsmd_f)-1, m=200, pl=F)
  cal_other_gen <- rms::val.prob(p=other_gen$PI, y=as.numeric(other_gen$chrsmd_f)-1, m=200, pl=F)
  
  c_stat_nested_ethnicity <- rbind(c_stat_nested_ethnicity, data.frame(
    C_test = c(concordance(m.white)$concordance,concordance(m.black)$concordance,concordance(m.asian)$concordance,concordance(m.mixed)$concordance,concordance(m.other_eth)$concordance),
    SE_test = c(concordance(m.white)$cvar,concordance(m.black)$cvar,concordance(m.asian)$cvar,concordance(m.mixed)$cvar,concordance(m.other_eth)$cvar),
    Fold = i,
    OuterRepeat = outer_rep,
    Ethnicity = c("White","Black","Asian","Mixed","Other"),
    n_test = c(nrow(white),nrow(black),nrow(asian),nrow(mixed),nrow(other_eth)),
    events_test = c(sum(white$chrsmd=="Yes"),sum(black$chrsmd=="Yes"),sum(asian$chrsmd=="Yes"),sum(mixed$chrsmd=="Yes"),sum(other_eth$chrsmd=="Yes")),
    intercept = c(unname(cal_white[12]),unname(cal_black[12]),unname(cal_asian[12]),unname(cal_mixed[12]),unname(cal_other_eth[12])),
    slope = c(unname(cal_white[13]),unname(cal_black[13]),unname(cal_asian[13]),unname(cal_mixed[13]),unname(cal_other_eth[13]))
  ))
  
  c_stat_nested_gender <- rbind(c_stat_nested_gender, data.frame(
    C_test = c(concordance(m.male)$concordance,concordance(m.female)$concordance,concordance(m.other_gen)$concordance),
    SE_test = c(concordance(m.male)$cvar,concordance(m.female)$cvar,concordance(m.other_gen)$cvar),
    Fold = i,
    OuterRepeat = outer_rep,
    Gender = c("Male","Female","Other"),
    n_test = c(nrow(male),nrow(female),nrow(other_gen)),
    events_test = c(sum(male$chrsmd=="Yes"),sum(female$chrsmd=="Yes"),sum(other_gen$chrsmd=="Yes")),
    intercept = c(unname(cal_male[12]),unname(cal_female[12]),unname(cal_other_gen[12])),
    slope = c(unname(cal_male[13]),unname(cal_female[13]),unname(cal_other_gen[13]))
  ))
  
}

  # Save out results
  list(
    best_inner_result_list = best_inner_result_list,
    best_lambda_list = best_lambda_list,
    c_stat_nested = c_stat_nested,
    coef = coef_df,
    c_stat_nested_ethnicity = c_stat_nested_ethnicity,
    c_stat_nested_gender = c_stat_nested_gender)
  
}

# Create a function to perform repeated nested cross-validation for random forest
repeated_nested_cv_rf <- function(combined_df, outerFolds, outerRepeats, innerFolds, innerRepeats, tuneGrid, seed) {
  
  set.seed(seed)
  seeds <- sample(1:10000, outerRepeats) # Generate unique seeds for each outer repeat
  
  # Create empty data frames to save out results
  best_inner_result_list <- list()
  best_mtry_list <- list()
  best_ntree_list <- list()
  best_nodesize_list <- list()
  all_inner_results <- list()
  c_stat_nested <- data.frame()
  calibration_slopes <- data.frame()
  var_imp_df <- data.frame()
  c_stat_nested_ethnicity <- data.frame()
  c_stat_nested_gender <- data.frame()
  
  # Outer loop for cross-validation
  for (outer_rep in 1:outerRepeats) {
    cat("Outer Repeat:", outer_rep, "\n")
    set.seed(seeds[outer_rep]) # Set a unique seed for each outer repeat
    labelled <- combined_df %>% filter(!is.na(chrsmd))
    unlabelled <- combined_df %>% filter(is.na(chrsmd))
    outer_folds <- createFolds(labelled$chrsmd, k = outerFolds, list = TRUE, returnTrain = TRUE)
    
    for (i in seq_along(outer_folds)) {
      cat("  Outer Fold:", i, "\n")
      labelled <- combined_df %>% filter(!is.na(chrsmd))
      unlabelled <- combined_df %>% filter(is.na(chrsmd))
      train_indices <- outer_folds[[i]]
      train <- labelled[train_indices, ] #  This splits the data into train (inner loop)
      test <- labelled[-train_indices, ] #  This splits the data into test (outer loop)
      
      # Impute test dataset
      mice_imputes_test = mice(test, m = 5, maxit = 5, seed = 12345, print=FALSE)
      test <- complete(mice_imputes_test)
      
      # Impute unlabelled data except for outcome
      mice_imputes_unlabelled = mice(unlabelled[,c(1:9)], m = 5, maxit = 5, seed = 12345, print=FALSE)
      unlabelled[,c(1:9)] <- complete(mice_imputes_unlabelled)
      
      # Inner Cross-Validation and Model Training
      for (ssl in 1:10){     
        cat("    Semi-Supervised Learning Iteration:", ssl, "\n")

      inner_results <- vector("list", innerRepeats) # Initialize as a list to store results for each inner repeat

      # Impute missing training data
      mice_imputes_train = mice(train, m = 50, maxit = 50, seed = 12345, print=FALSE)
      train <- complete(mice_imputes_train)
      
      for (inner_rep in 1:innerRepeats) {
        cat("    Inner Repeat:", inner_rep, "\n")
        
        # Use a different seed for each inner repeat
        set.seed(seeds[outer_rep] + inner_rep)
        
        # Tune hyperparameters
        for (mtry_value in tuneGrid) {
          library(randomForest)
          library(MLmetrics)
          library(caret)
          
          # Define a custom summary function to calculate F1 score
          customSummary <- function(data, lev = NULL, model = NULL) {
            f1 <- F1_Score(y_pred = data$pred, y_true = data$obs, positive = lev[1])
            out <- c(F1 = f1)
            out
          }
          
          # Set up the trainControl with the custom summary function
          control <- trainControl(method = 'cv', 
                                  number = 5, 
                                  #repeats = 10,
                                  classProbs = TRUE,
                                  summaryFunction = customSummary,
                                  search = 'random')
          
          tree <- c(50, 100, 250, 500)
          n.tree <- sample(tree,1)
          nodeSize <- seq(1,(nrow(train)/10), by=1)
          node.size <- sample(nodeSize,1)
          tune_grid_temp <- data.frame(mtry=c(NA,NA,NA))
          tune_grid_temp$mtry <- sample(tuneGrid$mtry,3)
          
          # Train the model using the custom F1 metric
          inner_model <- train(chrsmd ~ .,
                               data = train,
                               method = "rf",
                               metric = "F1",
                               tuneGrid = tune_grid_temp,
                               tuneLength=10,
                               ntree = n.tree,
                               nodesize=node.size,
                               trControl = control)
          
          print(inner_model)
          
          if (all(!is.na(inner_model$results$F1))){
            # Store the F1 for each inner repeat
            inner_results[[inner_rep]] <- rbind(inner_results[[inner_rep]], 
                                                data.frame(min.node.size = node.size,
                                                           tree = n.tree,
                                                           mtry = inner_model$results$mtry, 
                                                           F1 = inner_model$results$F1, 
                                                           repeat_number = inner_rep,
                                                           ssl = ssl))
          }
        }
      }
      
      # Combine results from all repeats
      all_inner_results <- do.call(rbind, inner_results)
      print(dim(all_inner_results))
      
      # Calculate the average F1 for each hyperparameter combination
      avg_results <- all_inner_results %>%
        group_by(mtry) %>%
        summarise(avg_F1 = mean(F1, na.rm=TRUE),
                  mtry = mean(mtry, na.rm=TRUE),
                  ntree = mean(tree, na.rm=TRUE),
                  nodesize=mean(min.node.size, na.rm=TRUE),
                  .groups = "keep") 
      
      # Determine the best hyperparameter combination based on the highest average F1
      best_inner_result <- avg_results[which.max(avg_results$avg_F1), ]
      best_min.node.size <- best_inner_result$nodesize
      best_mtry <- best_inner_result$mtry
      best_F1 <- best_inner_result$avg_F1
      best_ntree <- best_inner_result$ntree
      
      # Calculate the correct index for the current outer fold and repeat
      index <- (outer_rep - 1) * outerFolds + i
      
      # Store the best hyperparameters for the current outer fold
      best_inner_result_list[[index]] <- best_inner_result
      best_mtry_list[[index]]  <- best_mtry
      best_ntree_list[[index]]  <- best_ntree
      best_nodesize_list[[index]]  <- best_min.node.size
      
      # Train the final model using the best hyperparameters on the inner training set (x_inner_train, y_inner_train)
      control_final <- trainControl(method = 'none')
      repGrid <- data.frame(mtry=best_mtry)  
      
      best_inner_model <- train(chrsmd ~ .,
                          data = train,
                          method = "rf",
                          metric = "F1",
                          ntree = best_ntree,
                          nodesize=best_min.node.size,
                          trControl=control_final,
                          tuneGrid = repGrid)
      unlabelled <- unlabelled %>% mutate(PI = predict(best_inner_model, newdata = unlabelled, type = "prob")[,2],
                                          confident = case_when(PI>0.7 | PI<0.3 ~ 1,
                                                                TRUE ~ 0),
                                          chrsmd = case_when(PI>0.7 ~ "Yes",
                                                             PI<0.3 ~ "No"))
      confident_pseudolabels <- unlabelled %>% filter(confident==1)
      confident_pseudolabels <- confident_pseudolabels %>% subset(select=c(-confident, -PI))
      unlabelled <- unlabelled %>% filter(confident==0)
      
      if (nrow(confident_pseudolabels)<10) {
        print(paste0('Minimal confident pseudolabels after ', ssl, ' iterations'))
        break
      }
      
      train <- rbind(train, confident_pseudolabels)
      }
    }
      
      final_model <- train(chrsmd ~ .,
                           data = train,
                           method = "rf",
                           metric = "F1",
                           ntree = best_ntree,
                           nodesize=best_min.node.size,
                           trControl=control_final,
                           tuneGrid = repGrid)
      
      # Extract variable importance
      var_imp <- varImp(final_model)
      new_var_imp_df <- as.data.frame(var_imp$importance)
      new_var_imp_df$Predictor <- rownames(new_var_imp_df)
      new_var_imp_df <- new_var_imp_df[, c("Predictor", names(new_var_imp_df)[-ncol(new_var_imp_df)])]
      var_imp_df <- rbind(var_imp_df, new_var_imp_df)
      
      # Predict the linear predictors (PI) from the RF model
      test$PI <- predict(final_model, newdata = test, type = "prob")[,2]
      test$chrsmd_pred <- predict(final_model, newdata = test, type = "raw")
      test$chrsmd_f <- factor(test$chrsmd)
      
      cm <- confusionMatrix(data = test$chrsmd_pred, reference = test$chrsmd_f)
      test <- test %>% mutate(PI=case_when(PI==0 ~ 0.001, PI==1 ~ 0.999, TRUE ~ PI))
      
      # Fit a model using the predictors on the test set 
      model_test <- glm(chrsmd_f ~ PI, data = test, family="binomial")
      
      calibration <- rms::val.prob(p=test$PI, y=as.numeric(test$chrsmd_f)-1, m=200, pl=F)
      
      # Calculate calibration slope on the hold-out data (test set)
      
      calibration_intercept <- unname(calibration[12])
      calibration_slope <- unname(calibration[13])
      brier <- unname(calibration[11])
      
      c_stat_nested <- rbind(c_stat_nested, data.frame(
        C_test = concordance(model_test)$concordance,
        SE_test = concordance(model_test)$cvar,
        Fold = i,
        OuterRepeat = outer_rep,
        n_train = nrow(train),
        events_train = sum(train$chrsmd=="Yes"),
        n_test = nrow(test),
        events_test = sum(test$chrsmd=="Yes"),
        balanced_accuracy = cm$byClass[11],
        sensitivity = cm$byClass[1],
        specificity = cm$byClass[2],
        ppv = cm$byClass[3],
        npv = cm$byClass[4],
        precision = cm$byClass[5],
        recall = cm$byClass[6],
        f1 = cm$byClass[7],
        intercept = calibration_intercept,
        slope = calibration_slope,
        brier = brier
      ))
      
      # Test algorithmic bias
      white <- test %>% filter(ethnicity=="White")
      black <- test %>% filter(ethnicity=="Black")
      asian <- test %>% filter(ethnicity=="Asian")
      mixed <- test %>% filter(ethnicity=="Mixed")
      other_eth <- test %>% filter(ethnicity=="Other")
      male <- test %>% filter(gender=="Male")
      female <- test %>% filter(gender=="Female")
      other_gen <- test %>% filter(gender=="Other")
      
      m.white <- glm(chrsmd_f ~ PI, data = white, family="binomial")
      m.black <- glm(chrsmd_f ~ PI, data = black, family="binomial")
      m.asian <- glm(chrsmd_f ~ PI, data = asian, family="binomial")
      m.mixed <- glm(chrsmd_f ~ PI, data = mixed, family="binomial")
      m.other_eth <- glm(chrsmd_f ~ PI, data = other_eth, family="binomial")
      m.male <- glm(chrsmd_f ~ PI, data = male, family="binomial")
      m.female <- glm(chrsmd_f ~ PI, data = female, family="binomial")
      m.other_gen <- glm(chrsmd_f ~ PI, data = other_gen, family="binomial")
      
      cal_white <- rms::val.prob(p=white$PI, y=as.numeric(white$chrsmd_f)-1, m=200, pl=F)
      cal_black <- rms::val.prob(p=black$PI, y=as.numeric(black$chrsmd_f)-1, m=200, pl=F)
      cal_asian <- rms::val.prob(p=asian$PI, y=as.numeric(asian$chrsmd_f)-1, m=200, pl=F)
      cal_mixed <- rms::val.prob(p=mixed$PI, y=as.numeric(mixed$chrsmd_f)-1, m=200, pl=F)
      cal_other_eth <- rms::val.prob(p=other_eth$PI, y=as.numeric(other_eth$chrsmd_f)-1, m=200, pl=F)
      cal_male <- rms::val.prob(p=male$PI, y=as.numeric(male$chrsmd_f)-1, m=200, pl=F)
      cal_female <- rms::val.prob(p=female$PI, y=as.numeric(female$chrsmd_f)-1, m=200, pl=F)
      cal_other_gen <- rms::val.prob(p=other_gen$PI, y=as.numeric(other_gen$chrsmd_f)-1, m=200, pl=F)
      
      c_stat_nested_ethnicity <- rbind(c_stat_nested_ethnicity, data.frame(
        C_test = c(concordance(m.white)$concordance,concordance(m.black)$concordance,concordance(m.asian)$concordance,concordance(m.mixed)$concordance,concordance(m.other_eth)$concordance),
        SE_test = c(concordance(m.white)$cvar,concordance(m.black)$cvar,concordance(m.asian)$cvar,concordance(m.mixed)$cvar,concordance(m.other_eth)$cvar),
        Fold = i,
        OuterRepeat = outer_rep,
        Ethnicity = c("White","Black","Asian","Mixed","Other"),
        n_test = c(nrow(white),nrow(black),nrow(asian),nrow(mixed),nrow(other_eth)),
        events_test = c(sum(white$chrsmd=="Yes"),sum(black$chrsmd=="Yes"),sum(asian$chrsmd=="Yes"),sum(mixed$chrsmd=="Yes"),sum(other_eth$chrsmd=="Yes")),
        intercept = c(unname(cal_white[12]),unname(cal_black[12]),unname(cal_asian[12]),unname(cal_mixed[12]),unname(cal_other_eth[12])),
        slope = c(unname(cal_white[13]),unname(cal_black[13]),unname(cal_asian[13]),unname(cal_mixed[13]),unname(cal_other_eth[13]))
      ))
      
      c_stat_nested_gender <- rbind(c_stat_nested_gender, data.frame(
        C_test = c(concordance(m.male)$concordance,concordance(m.female)$concordance,concordance(m.other_gen)$concordance),
        SE_test = c(concordance(m.male)$cvar,concordance(m.female)$cvar,concordance(m.other_gen)$cvar),
        Fold = i,
        OuterRepeat = outer_rep,
        Gender = c("Male","Female","Other"),
        n_test = c(nrow(male),nrow(female),nrow(other_gen)),
        events_test = c(sum(male$chrsmd=="Yes"),sum(female$chrsmd=="Yes"),sum(other_gen$chrsmd=="Yes")),
        intercept = c(unname(cal_male[12]),unname(cal_female[12]),unname(cal_other_gen[12])),
        slope = c(unname(cal_male[13]),unname(cal_female[13]),unname(cal_other_gen[13]))
      ))
      
  
  }
  
  # Save out results
  list(
    best_inner_result_list = best_inner_result_list,
    best_mtry_list = best_mtry_list,
    best_ntree_list = best_ntree_list,
    best_nodesize_list = best_nodesize_list,
    c_stat_nested = c_stat_nested,
    var_imp_df = var_imp_df,
    c_stat_nested_ethnicity = c_stat_nested_ethnicity,
    c_stat_nested_gender = c_stat_nested_gender
    )
}

##### Run repeated nested CV for logistic regression and random forest ######

synth_enter <- synth[, grep("^enter", names(synth))]
synth_enter$age <- synth$age
synth_enter$gender <- synth$gender
synth_enter$ethnicity <- synth$ethnicity
synth_enter$chrsmd <- synth$chrsmd

# Define tuning grid for logistic regression
tune_grid_lr <- expand.grid(alpha = 1,  # LASSO (alpha = 1)
                         lambda = seq(0.0001, 1, length = 100))

# Run logistic regression in repeated nested CV
results_lr_enter <- repeated_nested_cv_lr(
  combined_df = synth_enter, 
  outerFolds = outerFolds, 
  outerRepeats = outerRepeats, 
  innerFolds = innerFolds, 
  innerRepeats = innerRepeats, 
  tuneGrid = tune_grid_lr,
  seed = 231
)

# Run logistic regression in repeated nested CV
results_lr <- repeated_nested_cv_lr(
  combined_df = synth, 
  outerFolds = outerFolds, 
  outerRepeats = outerRepeats, 
  innerFolds = innerFolds, 
  innerRepeats = innerRepeats, 
  tuneGrid = tune_grid_lr,
  seed = 231
)

coef_summary <- results_lr_enter$coef %>% aggregate(Coefficient ~ Feature, FUN="mean")
write.csv(coef_summary, "Results/2. Variable Importance/lr_enter_coef.csv", row.names = TRUE)

write_csv(results_lr_enter$best_inner_result_list, "Results/1. Performance/best_inner_result_lr_enter.csv")
write_csv(results_lr_enter$best_lambda_list, "Results/1. Performance/best_lambda_lr_enter.csv")
write_csv(results_lr_enter$c_stat_nested, "Results/1. Performance/performance_lr_enter.csv")
write_csv(results_lr_enter$c_stat_nested_ethnicity, "Results/1. Performance/ethnicity_bias_lr_enter.csv")
write_csv(results_lr_enter$c_stat_nested_gender, "Results/1. Performance/gender_bias_lr_enter.csv")

coef_summary <- results_lr %>% aggregate(Coefficient ~ Feature, FUN="mean")
write.csv(coef_summary, "Results/2. Variable Importance/lr_coef.csv", row.names = TRUE)

write_csv(results_lr$best_inner_result_list, "Results/1. Performance/best_inner_result_lr.csv")
write_csv(results_lr$best_lambda_list, "Results/1. Performance/best_lambda_lr.csv")
write_csv(results_lr$c_stat_nested, "Results/1. Performance/performance_lr.csv")
write_csv(results_lr$c_stat_nested_ethnicity, "Results/1. Performance/ethnicity_bias_lr.csv")
write_csv(results_lr$c_stat_nested_gender, "Results/1. Performance/gender_bias_lr.csv")

# Define tuning grid for random forest
tune_grid_rf <- expand.grid(
  mtry=c(3:30)
)

# Run random forest in repeated nested CV
results_rf_enter <- repeated_nested_cv_rf(
  combined_df = synth_enter, 
  outerFolds = outerFolds, 
  outerRepeats = outerRepeats, 
  innerFolds = innerFolds, 
  innerRepeats = innerRepeats, 
  tuneGrid = tune_grid_rf,
  seed = 231
)

# Run random forest in repeated nested CV
results_rf <- repeated_nested_cv_rf(
  combined_df = synth, 
  outerFolds = outerFolds, 
  outerRepeats = outerRepeats, 
  innerFolds = innerFolds, 
  innerRepeats = innerRepeats, 
  tuneGrid = tune_grid_rf,
  seed = 231
)

var_imp_summary <- results_rf_enter$var_imp_df %>% aggregate(Overall ~ Predictor, FUN="mean")
write.csv(var_imp_summary, "Results/2. Variable Importance/rf_enter_variable_importance.csv", row.names = TRUE)

write_csv(results_rf_enter$best_inner_result_list, "Results/1. Performance/best_inner_result_rf_enter.csv")
write_csv(results_rf_enter$best_mtry_list, "Results/1. Performance/best_mtry_rf_enter.csv")
write_csv(results_rf_enter$best_ntree_list, "Results/1. Performance/best_ntree_rf_enter.csv")
write_csv(results_rf_enter$best_nodesize_list, "Results/1. Performance/best_nodesize_rf_enter.csv")
write_csv(results_rf_enter$c_stat_nested, "Results/1. Performance/performance_rf_enter.csv")
write_csv(results_rf_enter$c_stat_nested_ethnicity, "Results/1. Performance/ethnicity_bias_rf_enter.csv")
write_csv(results_rf_enter$c_stat_nested_gender, "Results/1. Performance/gender_bias_rf_enter.csv")

var_imp_summary <- results_rf %>% aggregate(Overall ~ Predictor, FUN="mean")
write.csv(var_imp_summary, "Results/2. Variable Importance/rf_variable_importance.csv", row.names = TRUE)

write_csv(results_rf$best_inner_result_list, "Results/1. Performance/best_inner_result_rf.csv")
write_csv(results_rf$best_mtry_list, "Results/1. Performance/best_mtry_rf.csv")
write_csv(results_rf$best_ntree_list, "Results/1. Performance/best_ntree_rf.csv")
write_csv(results_rf$best_nodesize_list, "Results/1. Performance/best_nodesize_rf.csv")
write_csv(results_rf$c_stat_nested, "Results/1. Performance/performance_rf.csv")
write_csv(results_rf$c_stat_nested_ethnicity, "Results/1. Performance/ethnicity_bias_rf.csv")
write_csv(results_rf$c_stat_nested_gender, "Results/1. Performance/gender_bias_rf.csv")

##### Compile algorithmic bias results #####
# LR ENTER
m.eth.C.lr_enter <- aov(C_test ~ Ethnicity, data=results_lr_enter$c_stat_nested_ethnicity)
bias.eth.C.lr_enter <- effectsize::cohens_f(m.eth.C.lr_enter)

m.gen.C.lr_enter <- aov(C_test ~ Gender, data=results_lr_enter$c_stat_nested_gender)
bias.gen.C.lr_enter <- effectsize::cohens_f(m.gen.C.lr_enter)

m.eth.slope.lr_enter <- aov(slope ~ Ethnicity, data=results_lr_enter$c_stat_nested_ethnicity)
bias.eth.slope.lr_enter <- effectsize::cohens_f(m.eth.slope.lr_enter)

m.gen.slope.lr_enter <- aov(slope ~ Gender, data=results_lr_enter$c_stat_nested_gender)
bias.gen.slope.lr_enter <- effectsize::cohens_f(m.gen.slope.lr_enter)

mean_eth_bias_lr_enter <- mean(bias.eth.slope.lr_enter,bias.eth.C.lr_enter)
mean_gen_bias_lr_enter <- mean(bias.gen.slope.lr_enter,bias.gen.C.lr_enter)

# LR All
m.eth.C.lr <- aov(C_test ~ Ethnicity, data=results_lr$c_stat_nested_ethnicity)
bias.eth.C.lr <- effectsize::cohens_f(m.eth.C.lr)

m.gen.C.lr <- aov(C_test ~ Gender, data=results_lr$c_stat_nested_gender)
bias.gen.C.lr <- effectsize::cohens_f(m.gen.C.lr)

m.eth.slope.lr <- aov(slope ~ Ethnicity, data=results_lr$c_stat_nested_ethnicity)
bias.eth.slope.lr <- effectsize::cohens_f(m.eth.slope.lr)

m.gen.slope.lr <- aov(slope ~ Gender, data=results_lr$c_stat_nested_gender)
bias.gen.slope.lr <- effectsize::cohens_f(m.gen.slope.lr)

mean_eth_bias_lr <- mean(bias.eth.slope.lr,bias.eth.C.lr)
mean_gen_bias_lr <- mean(bias.gen.slope.lr,bias.gen.C.lr)

# RF ENTER
m.eth.C.rf_enter <- aov(C_test ~ Ethnicity, data=results_rf_enter$c_stat_nested_ethnicity)
bias.eth.C.rf_enter <- effectsize::cohens_f(m.eth.C.rf_enter)

m.gen.C.rf_enter <- aov(C_test ~ Gender, data=results_rf_enter$c_stat_nested_gender)
bias.gen.C.rf_enter <- effectsize::cohens_f(m.gen.C.rf_enter)

m.eth.slope.rf_enter <- aov(slope ~ Ethnicity, data=results_rf_enter$c_stat_nested_ethnicity)
bias.eth.slope.rf_enter <- effectsize::cohens_f(m.eth.slope.rf_enter)

m.gen.slope.rf_enter <- aov(slope ~ Gender, data=results_rf_enter$c_stat_nested_gender)
bias.gen.slope.rf_enter <- effectsize::cohens_f(m.gen.slope.rf_enter)

mean_eth_bias_rf_enter <- mean(bias.eth.slope.rf_enter,bias.eth.C.rf_enter)
mean_gen_bias_rf_enter <- mean(bias.gen.slope.rf_enter,bias.gen.C.rf_enter)

# RF All
m.eth.C.rf <- aov(C_test ~ Ethnicity, data=results_rf$c_stat_nested_ethnicity)
bias.eth.C.rf <- effectsize::cohens_f(m.eth.C.rf)

m.gen.C.rf <- aov(C_test ~ Gender, data=results_rf$c_stat_nested_gender)
bias.gen.C.rf <- effectsize::cohens_f(m.gen.C.rf)

m.eth.slope.rf <- aov(slope ~ Ethnicity, data=results_rf$c_stat_nested_ethnicity)
bias.eth.slope.rf <- effectsize::cohens_f(m.eth.slope.rf)

m.gen.slope.rf <- aov(slope ~ Gender, data=results_rf$c_stat_nested_gender)
bias.gen.slope.rf <- effectsize::cohens_f(m.gen.slope.rf)

mean_eth_bias_rf <- mean(bias.eth.slope.rf,bias.eth.C.rf)
mean_gen_bias_rf <- mean(bias.gen.slope.rf,bias.gen.C.rf)

##### Evaluate models #####
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
eval <- data.frame(model=c("LR - ENTER", "LR - All", "RF - ENTER", "RF - All"),
                   C=c(mean(normalize(results_lr_enter$c_stat_nested$C_test)),mean(normalize(results_lr$c_stat_nested$C_test)), mean(normalize(results_rf_enter$c_stat_nested$C_test)),mean(normalize(results_rf$c_stat_nested$C_test))),
                   calibration=c(mean(normalize(results_lr_enter$c_stat_nested$slope)),mean(normalize(results_lr$c_stat_nested$slope)), mean(normalize(results_rf_enter$c_stat_nested$slope)),mean(normalize(results_rf$c_stat_nested$slope))),
                   bias.eth=c(mean_eth_bias_lr_enter, mean_eth_bias_lr,mean_eth_bias_rf_enter,mean_eth_bias_rf), # Algorithmic fairness assessment - ethnicity
                   bias.gen=c(mean_gen_bias_lr_enter, mean_gen_bias_lr,mean_gen_bias_rf_enter,mean_gen_bias_rf), # Algorithmic fairness assessment - gender
                   complexity=c(1,1,2,2), # LR is less complex
                   SU=c(5,5,4,4), # Slight preference for LR
                   PPI=c(5,5,4,4) # Slight preference for LR
                   )

# Reverse score complexity and bias scores so that higher values=better for all measures
eval$complexity_rev <- 1 / eval$complexity
eval$bias_eth_rev <- 1 / eval$bias_eth
eval$bias_gen_rev <- 1 / eval$bias_gen

eval_norm <- eval %>%
  mutate(across(c(bias_eth_rev, bias_gen_rev, complexity_rev, SU, PPI), normalize))

eval_norm$composite_score <- with(eval_norm, 
                             0.2 * C + 
                               0.2 * calibration + 
                               0.1 * bias_eth_rev + 
                               0.1 * bias_gen_rev + 
                               0.1 * complexity_rev + 
                               0.1 * SU + 
                               0.2 * PPI)

eval_norm <- eval_norm %>%
  arrange(desc(composite_score))

write_csv(eval_norm,"/Results/3. Model Evaluation/Model_Evaluation.csv")

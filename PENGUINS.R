#####Machine learning######
library(palmerpenguins)
library(xgboost)
library(doParallel)
library(tidyverse)
library(tidymodels)
library(nnet)
library(cowplot)

# Load in the packages required for this adventure
library(tidyverse)
library(palmerpenguins)
library(here)

# View the first 10 rows of the data
penguins %>% 
  slice_head(n = 10)

library(cowplot)
ggdraw() + 
  draw_image("culmen_depth.png", width = 0.5) + 
  draw_image("peng.jpg", width = 0.5, x = 0.5)


## The plan

#1.  Exploratory Data Analysis
#2.  Preprocess data with recipes
#3.  Create 2 model specifications
#4.  Resampling to tune and compare model performance
#5.  The last fit
#6.  Model inferencing

#EDA
## Sum of missing values across the columns
colSums(is.na(penguins)) %>% 
  as.data.frame()

# Where are those missing values?
penguins %>% 
  filter(if_any(everything(), ~is.na(.x)))

# Set theme
theme_set(theme_light())
penguins <- as.data.frame(penguins)
# Pivot data to a long format
# Pivot data to a long format
# Pivot data to a long format
penguins_select_long <- penguins %>%
  select(where(is.numeric), species) %>% 
  pivot_longer(!species, names_to = "predictors", values_to = "values")

str(penguins)
penguins_select_long %>% 
  slice_sample(n = 10)

# Make a box plot for each predictor 
theme_set(theme_light())
penguins_select_long %>%
  ggplot(mapping = aes(x = species, y = values)) +
  geom_jitter(aes(color = species),
              width = 0.1,
              alpha = 0.7,
              show.legend = FALSE) +
  geom_boxplot(aes(color = species, fill = species), width = 0.25, alpha = 0.3) +
  facet_wrap(~predictors, scales = "free") +
  scale_color_manual(values = c("darkorange","darkorchid","cyan4")) +
  scale_fill_manual(values = c("darkorange","darkorchid","cyan4")) +
  theme(legend.position = "none")

#### Does island of origin matter?
#Starting with `penguin` data, `filter` to obtain observations where `sex` is not NA, AND THEN, `count` the sex in each island.
penguins %>%
  count(island, species)

# Make a bar plot representing the count of penguins in each island
ggplot(penguins, aes(x = island, fill = species)) +
  geom_bar(alpha = 0.8) +
  scale_fill_manual(values = c("darkorange","purple","cyan4"),
                    guide = FALSE) +
  theme_minimal() +
  facet_wrap(~species, ncol = 1) +
  coord_flip()


# Drop the year and island columns
df_penguins <- penguins %>% 
  select(-island, -sex)

# View first 10 observations
df_penguins %>% 
  slice_head(n = 10)

##Modeling
set.seed(123)
library(tidymodels)

# Create a data split specification with 75% of data
# used as train set and the rest as test set
penguins_split <- initial_split(df_penguins,
                                prop = 0.75,
                                strata = species)

# Extract the training and testing set
penguins_train <- training(penguins_split)
penguins_test <- testing(penguins_split)


# Create preprocessing recipe
penguins_rec <- recipe(species ~ ., data = penguins_train) %>% 
  # Impute numeric variables using mean
  step_impute_mean(all_numeric_predictors()) %>% 
  # Normalize numeric variables
  step_normalize(all_numeric_predictors())



# Print recipe
penguins_rec

# Summary of variables in recipe
summary(penguins_rec)


# Model **type** differentiates models such as logistic regression, decision tree models, and so forth.

# Model **engine** is the computational tool which will be used to fit the model. Often these are R packages, such as **`"lm"`** or **`"ranger"`**
  
# Model **mode** includes common options like regression and classification; some model types support either of these while some only have one mode.


#We'll create 2 types of models:

#A Multinomial regression model: extension of binomial logistic regression to predict multiclass data

#A boosted tree model: Ensemble of trees `built` **sequentially** on the `outputs` of the `previous one` in an attempt to incrementally reduce the *loss* (error) in the model.

# Create a multinomial regression model specification
mlr_spec <- 
  # Type of model
  multinom_reg(penalty = tune()) %>% 
  # Engine
  set_engine("nnet") %>% 
  # Mode
  set_mode("classification")

# Boosted tree regression model specification
boost_spec <- boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

# Logistic regression workflow
mlr_wf <- workflow() %>% 
  add_recipe(penguins_rec) %>% 
  add_model(mlr_spec)

mlr_wf

# xgboost workflow
boost_wf <- workflow() %>% 
  add_recipe(penguins_rec) %>% 
  add_model(boost_spec)

boost_wf

#Time to train some models!!
### 4. Comparing and tuning models with Resamples

#Once you create two or more models, the next step is to compare them. But should you re-predict the training set and compare model performance? How do we deal with model tuning like in this example?
#The main take-away from this example is that re-predicting the training set is a **bad idea** for most models. If the test set should not be used immediately, and re-predicting the training set is a bad idea, what should be done?
#*Resampling methods*, such as bootsrap resampling, cross-validation or validation sets, are the solution.
#The idea of resampling is to create simulated data sets that can be used to estimate the performance of your models or tune model hyperparameters, say, because you want to compare models.

#### 
# Create train resample on the TRAINING SET
set.seed(123)
penguins_boot <- bootstraps(data = penguins_train)
head(penguins_boot)


#Now let's compare our models. We'll start by fitting our boosted trees model to the 25 simulated analysis data sets and evaluate how it performs on the corresponding assessment sets. The final estimate of performance for the model
#is the average of the 25 replicates of the statistics.

# Fit boosted trees model to the resamples
boost_rs <- fit_resamples(
  object = boost_wf,
  resamples = penguins_boot,
  metrics = metric_set(accuracy)
)


# Show the model with best metrics
show_best(boost_rs) %>% 
  as.data.frame()

#Let's do the same for our multinomial model. Remember, it has a **tuning** parameter.
#We can *estimate* the *best values* for these by training many models on resamples and 
#measuring how well all these models perform. 
#This process is called **tuning**.
#Tidymodels provides a way to *tune* hyperparameters by trying multiple combinations and 
#finding the best result for a given performance metric. That means, we need a set of
#possible values for each parameter to try. In this case study, we'll work through a regular 
#grid of hyperparameter values.

# Create a grid of tuning parameters
grid <- grid_regular(penalty(),
                     levels = 10)

# Display some of the penalty values that will be used for tuning 
grid

set.seed(2056)
doParallel::registerDoParallel()

# Tune the value for penalty
mlr_tune <- tune_grid(
  object = mlr_wf,
  resamples = penguins_boot,
  metrics = metric_set(accuracy),
  grid = grid
)

# Show model with best metrics
show_best(mlr_tune) %>% as.data.frame()

# Initial workflow
mlr_wf

# Finalize the workflow
final_wf <- mlr_wf %>% 
  finalize_workflow(parameters = select_best(mlr_tune))

final_wf


# Make a last fit
final_fit <- final_wf %>% 
  last_fit(penguins_split)


# Collect metrics
final_fit %>% 
  collect_metrics()

#Perhaps explore other metrics such as a confusion matrix? A confusion matrix allows you to 
#compare the observed and predicted by tabulating how many examples in each class were correctly
#classified by a model.



# Create confusion matrix
collect_predictions(final_fit) %>% 
  conf_mat(truth = species, estimate = .pred_class)

# Visualize confusion matrix
collect_predictions(final_fit) %>% 
  conf_mat(truth = species, estimate = .pred_class) %>% 
  autoplot(type = "heatmap")

# Other metrics that arise from confusion matrix
collect_predictions(final_fit) %>% 
  conf_mat(truth = species, estimate = .pred_class) %>% 
  summary() %>% 
  filter(.metric %in% c("accuracy", "sens", "ppv", "f_meas"))


#🎓 Recall: defined as the proportion of positive results out of the number of samples which were actually positive. Also known as `sensitivity`.

#🎓 Specificity: defined as the proportion of negative results out of the number of samples which were actually negative.

#🎓 Accuracy: The percentage of labels predicted accurately for a sample.

#🎓 F Measure: A weighted average of the precision and recall, with best being 1 and worst being 0.

### 6. Use the model with new data observations
# Extract trained workflow
penguins_mlr_model <- final_fit %>% 
  extract_workflow()

# Save workflow
saveRDS(penguins_mlr_model, "penguins_mlr_model.rds")

# Load model
loaded_mlr_model <- readRDS("penguins_mlr_model.rds") 

# Create new tibble of observations
new_obs <- tibble(
  bill_length_mm = c(49.5, 38.2),
  bill_depth_mm = c(18.4, 20.1),
  flipper_length_mm = c(195, 190),
  body_mass_g = c(3600, 3900))
new_obs
# Make predictions
new_results <- new_obs %>% 
  bind_cols(loaded_mlr_model %>% 
              predict(new_data = new_obs))

# Show predictions
new_results













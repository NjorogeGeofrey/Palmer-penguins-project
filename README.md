# Machine Learning with Palmer Penguins
This project uses machine learning techniques to analyze the Palmer Penguins dataset. It includes exploratory data analysis (EDA), preprocessing steps, model building, tuning, and inferencing. Below is an overview of the project structure and key components.

## Installation
To run this project, you'll need to have R and the following R packages installed:

palmerpenguins
xgboost
doParallel
tidyverse
tidymodels
nnet
cowplot


You can install these packages using the following R commands:
## R code
install.packages(c("palmerpenguins", "xgboost", "doParallel", "tidyverse", "tidymodels", "nnet", "cowplot"))

## Usage
Clone this repository to your local machine.
Open the R script containing the project code.
Run the script to perform EDA, model training, and inferencing.

## Project Structure
Exploratory Data Analysis (EDA): The script includes code for exploring the dataset, handling missing values, and visualizing data distributions.
Preprocessing: Data preprocessing steps such as imputation and normalization are included using the tidymodels package.
Model Building: Two types of models are built: a multinomial regression model and a boosted tree model using xgboost.
Model Tuning: Hyperparameter tuning is performed using resampling methods and grid search.
Inferencing: The final trained model is used for making predictions on new data observations.

## How to Run
Ensure all required R packages are installed.
Open the R script in your R environment.
Run the script to execute the entire workflow.

## Additional Information
The project uses the Palmer Penguins dataset for training and testing.
Model performance metrics such as accuracy, sensitivity, specificity, and F measure are evaluated.
Confusion matrices and visualizations are used to analyze model predictions.

## Contact
For any questions or feedback regarding this project, please contact njorogeofrey73@gmail.com.



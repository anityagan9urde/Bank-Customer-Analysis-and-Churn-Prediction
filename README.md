# Bank-Customer-Analysis-and-Churn-Prediction
This is a Jupyter Notebook showing the Exploratory Data Analysis and the model making process for a Customer Dataset.<br>
[Show Notebook](https://github.com/AnityaGan9urde/Bank-Customer-Analysis-and-Churn-Prediction/blob/main/bank-customer-churn-prediction-xgboost-gpu.ipynb)
### Dataset: 
- I've done this analysis on a Kaggle Dataset which is available here: https://www.kaggle.com/shrutimechlearn/churn-modelling
- The dataset is a Bank Customer dataset with target values showing if that customer has left the bank or closed their accounts.
- **Features:**
  - CustomerId: Unique Ids for bank customer identification
  - Surname: Customer's last name
  - CreditScore: Credit score of the customer
  - Geography: The country from which the customer belongs
  - Gender: Male or Female
  - Age: Age of the customer
  - Tenure: Number of years for which the customer has been with the bank
  - Balance: Bank balance of the customer
  - NumOfProducts: Number of bank products the customer is utilising
  - HasCrCard: Binary Flag for whether the customer holds a credit card with the bank or not
  - IsActiveMember: Binary Flag for whether the customer is an active member with the bank or not
  - EstimatedSalary: Estimated salary of the customer in Dollars
- **Target:**
  - Exited: Binary flag 1 if the customer closed account with bank and 0 if the customer is retained

### Data Wrangling, Feature Engineering and EDA:
- The dataset had a total of 4 categorical features- *Geography, Gender, HasCrCard, IsActiveMember*; 6 numerical features- *CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary.*
- The columns- *CustomerId, Surname, Exited*, were omitted from the features as they did not contribute anything to the training process.
- After extensive Exploratory Data Analysis using **Matplotlib** and **Seaborn**, I decided to one hot encode the categorical features and divide the numerical features into seperate bands and then create separate categories for each band.
- For example, if the Age feature ranges from 18 to 60, then I will create 5 sections ranging from 18 to 26, 27 to 35 and so on. And all these sections will get a seperate category ranging from 1 to 5 in this case.
- I used a more raw approach while implementing one-hot and label encoding using the `.map()` function in python.
- I also found that the dataset was slightly imbalanced and there were more observations for the customers not leaving the bank. To be exact, just 25% of the observations belonged to class 1(i.e. Exited).
- Thus, my dataset was cleaned and ready to be passed into a model of my choice.
### Modelling:
- I used **Logistic Regression** as the base model to get a score of **0.75** on the test data.
- As Logistic Regression could not capture the underlying function, I used a more advanced model like **XG-Boost**, which is a boosting algorithm effective in understanding the more complex relationships between the features.
- This time I got a score of **0.81** on the test data.
- ### Hyper Parameter Tuning using **Grid Search CV** and using **Repeated Stratified KFold**:
  - I tweaked the hyperparameters for the XG-Boost algorithm and applied cross validation while training.
  - The hyperparameters were adjusted referencing the documentation and trial and error.
  - For cross-validation, Repeated Stratified KFold was used as a strategy because it helps if the dataset was imbalanced.
  - The Best Scores for different runs during training were as such:
    - Best Score: **85.40%** with these parameters: {'alpha': 0.01, 'eval_metric': 'auc', 'gamma': 0.2, 'learning_rate': 0.1, 'max_depth': 4, 'objective': 'binary:logistic', 'scale_pos_weight': 0.26, 'subsample': 0.5, 'tree_method': 'gpu_hist'}),
    - Best Score: **85.43%** with these parameters: {'alpha': 0.01, 'eval_metric': 'auc', 'gamma': 0.3, 'learning_rate': 0.1, 'max_depth': 4, 'objective': 'binary:logistic', 'scale_pos_weight': 0.26, 'subsample': 0.6, 'tree_method': 'gpu_hist'}),
    - Best Score: **85.44%** with these parameters: {'alpha': 0.01, 'eval_metric': 'auc', 'gamma': 0.2, 'learning_rate': 0.11, 'max_depth': 4, 'objective': 'binary:logistic', 'scale_pos_weight': 0.26, 'subsample': 0.7, 'tree_method': 'gpu_hist'}),
    - Best Score: **85.44%** with these parameters: {'alpha': 0.1, 'eval_metric': 'auc', 'gamma': 0.3, 'learning_rate': 0.11, 'max_depth': 4, 'objective': 'binary:logistic', 'scale_pos_weight': 0.26, 'subsample': 0.7, 'tree_method': 'gpu_hist'})

### Results:
- Plot for ROC-AUC curves for different models(columns), for training and testing datasets(rows):
![](https://github.com/AnityaGan9urde/Bank-Customer-Analysis-and-Churn-Prediction/blob/main/__results___91_2.jpg)<br>
- The above three plots give the ROC AUC curve for the training sets and the below three plots for the testing sets. It can be observed that the curve for training set is more smooth as compared to the curves for the testing sets.
- For the XG-Boost model trained without cross validation and hyperparameter tuning(middle column), the training AUC score is **0.95** but while testing it falls to **0.81**, suggesting that the model must have overfit to the data.
- In the third column, the XG-Boost model which used cross validation during training and was hyper tuned, the training score was **0.88** and the testing score was **0.84**, suggesting that the model did not overfit that much to the training data and hence the testing score was the highest among all three models.
#### Thus, we can observe that the AUC score improved substantially due to XG-Boost paired with the fine tuning and cross validation.


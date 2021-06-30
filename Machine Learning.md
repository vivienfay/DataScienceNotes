## 1. Overfitting

---

- Overfitting: the testing error is bigger than training error
- How to solve overfitting problem
  - Increasing training data set(also decrease the High Variance)
  - Avoid over-training your dataset:
    - Filter out features such as feature reduction, PCA（manually/ model selection）
    - Regularization(keep all features)
    - Ensemble Learning(bagging boosting)
- curse of dimensionality: As the number of dimension increase, the distribution of samples in space are more sparse

\
&nbsp;

## 2. Bias & Variance

---

- Bias: bias is the difference between the average prediction of our model and the correct value which we are trying to predict.
- Variance: is the variability of model prediction for a given data point or a value which tells us spread of our data

\
&nbsp;

## 3. Ensemble Learning

---

- Including Bagging and Boosting
- Bagging: more stable to avoid overfitting, decreasing the variance
- Boosting: change the distribution of samples. optimize the loss function to decrease the bias, they convert a set of weak learners into a single strong learners
- Difference:
  - Sample:
    - Bagging: Bootstrapping. With Replacement
    - Boosting: The proportion of samples in classifier has some changes
  - Proportion of  samples
    - bagging: The weight of samples are the same
    - boosting: Change the weight of samples based on error rate
  - Loss Function:
    - bagging: all the weight are the same
    - boosting: change the weight bas
  - Parallel:
    - bagging: can be parelleled
    - boosting: sequential
- Example in tree:
  - Random Forest:
  - Adaboost:
    - adaptive boosting changes the sample distribution by modifying the weights attached to each of the instances.
    - pros: no need to do feature selection / no worry to overfitting
    - cons: More sensitive to abnormal data and noise / computation is slow
  - Gradient boost: the weak learner trains on the remaining errors of the strong learners.
  - XGBOOST: XGBOOST used a more regularized model formalization to control over-fitting, and from the engineering perspectives, push the limits of computation resources for boosted tree algorithms

\
&nbsp;

## 4. Resampling Method

---

- Boostraping: random sampling with replacement
  - evaluating model performance, model selection (select the appropriate level of flexibility)
- Cross-Validation: random sampling with no replacement
  - mostly used to quantify the uncertainty associated with a given estimator or statistical learning method

\
&nbsp;

## 5. Linear Regression

---

- Assumption: 1. y obey Normal distribution 2. errors are independent 3. variable should be independent 4. The variance of errors are constant 5.errors are normal distribution
- Loss Function:
  
- Estimate the parameter:
  - OLS: minimize the loss function to estimate the parameter
  - gradient descent(learning rate, stepwise): used in iterative computation
  - Maximum Likelihood Estimation(used to get Beta): The most appropriate parameter makes the probability of extracting observation from the distribution largest.
- Model Assessment:
  - R-square: the proportion of variability in the response that is explained by the regression. [0,1], the closer to 1 it is the model fits better.
  - MSE= Mean square error
   
  - RMSE: Absolute measure of fit (whereas R2R2 is a relative measure of fit)
- pros:
  - easy to interpret
  - speed is fast


\
&nbsp;

## 6. Logistic Regression

---

- Assumption: target variable follows the bernoulli distribution
- Using sigmoid function to map the target variable to 0 or 1.

- Interpretation of the coefficients: the increase of logodds log⁡odds for the increase of one unit of a predictor, given all the other predictors are fixed.
- Maximum Liklihood Estimation

- cons:
  - need to convert non-linear feature
  - tend to be overfitting

\
&nbsp;

## 7. Multiple Regression

---

- Model Assessment:
  - Adjusted R-Square: when multiple variables, consider the degree of freedom

  - F test:(when using multiple variables)
    - Evaluate the hypothesisHo: all regression coefficients are equal to zero Vs H1H1: at least one doesn’t
    - Indicates that R2R2 is reliable

- Analysis of residuals:
  - QQ plot
  - Scatter plots residuals Vs predictors
  - Normality of errors

\
&nbsp;

## 8. Regularization

---

- Regularization:
  - Used to prevent overfitting: improve the generalization of a model
  - decrease the complexity of a model
  - add the penalty
- Ridge regression:(L2 penalty)
  - Loss function =
   
  - The overfitting reason is that there is a gap between testing error and training error. Testing error may have more variance. In order to consider the variance in testing error, penalty is introduced into loss function. It can control the model complexity.
  - Trade off the variance and bias through controlling hyper parameter (lambda), We can use cross validation to determine the value of lambda
  - a bit faster than the lasso
- Lasso regression: (L1 penalty)
  - Can force regression coefficients to be exactly 0: feature selection method by itself
  -
- Logistics regression - l1 & l2 regression

- The difference between lasso and ridge and elastic net:
  - Lasso can make the parameter with smaller value as 0. We can think it as a way to reduce the dimension or feature selection
  - Elastic net: if the feature number is bigger than training set or some features are highly correlated, it is more stable than lasso

\
&nbsp;

## 6. Evaluation

---

- Algorithm Perspective:
  - Complexity: Space / Time
  - Performance on a given dataset
  - Performance on several dataset
- Cross validation
  - Goal: Assess how your model result will generalize to another independent data set. estimate how your model will perform in practice
  - type:
    - K-fold:
      - do hyperparameter tuning
      -
    - Leave-one-out:
- Regression evaluation
  - RMSE: root mean of square error
  - MAE: Mean absolute error
- Classification evaluation
  - Confusion Matrix
  -
  - Metrics:
    - Precision = TP / (TP+FP) —查全率： how many sample which are predicted as positive is actually true
    - Recall (Sensitivity)=  TP / (TP+FN) — 查准率: how many actually positive sample are correctly predicted
    - F1= 2 / ( 1 / precision + 1 / recall)
    - Accuracy = ( TP + TN ) / (TP + TN + FP + FN )
    - Misclassification Rate
    
    - Example
      - Spam email: precision (you don’t want to miss the negative things)
      - Disease detection: recall (you don’t want to miss the positive things)
      - Anomaly detection: recall (you don’t want to miss the positive things)
- ROC Curve

 - X - axis: false positive rate = false positive / real negative
 - Y - axis: true positive rate = true positive / real positive = recall
 - Equal error rate = FPR = TPR
 - ROC is created by different threshold
 - Special points on ROC:
  - (0,0): threshold is the lowest
  - (1,1): threshold is the largest
  - (0,1): best case
  - (1,0): worst case

- AUC
  - The area under ROC curve, in [0,1], the larger AUC is, the Better the model performance is.
  - AUC is a probability value. It is the probability a randomly-chosen positive example is ranked more highly than a randomly-chossen negative example.
- PR Curve
  - better for imbalanced problem

\
&nbsp;

## 8. Decision Tree (<https://zhuanlan.zhihu.com/p/32053821>)

---

- How to build a decision tree?
  - Two steps: feature prioritization , Feature splitting(information gain)
  - Algorithm Details - ID3 Algorithm
    - At each iteration step, we split the feature which can give us the largest entropy gain.
      - entropy:
     
      - Entropy of entire information system:
     
  - Other algorithm
    - C4.5 - Gain Rate(the larger the better)
    - CART - Gini impurity(classification) LSD(regression)（the smaller the better）
  - Compare decision tree with linear regression
    - Decision tree’s formula can be like this:
    
    - Contribution has some connection with Vroot in nonlinear model, while the parameter won’t be affected by x in linear model.

\
&nbsp;

## 9. Random Forest

---

- Decision Tree + Bagging + feature sampling
  - Bagging: Sampling with replacement.
    - Bagging == Bootstrap aggregating
    - Bagging can decrease the variance by introducing randomness into your model framework
  - Feature sampling: randomly pick features in every tree
    - Generally floor(sqrt(ncol(x)))
  - Advantages of random forest
    - Less overfitting(Since decision tree often overfits if we don’t prune it)
    - Parallel implementation(also fast)
  - Feature importance
  -
    - It does not signify positive or negative impact on the class label. It only judges how much class discriminatory information each variable contains.
- Paramater: grid search, n estimators, criterion, max_leaf_nodes or max_depth, max features(sort()), criterion
- adaboost vs GBDT:
  - Adaboost update the classifier by changing the weight of data which has higher error rarte
  - GBDT: calculating the gradient for error
  - XGboost: based on GBDT: add the regularization. using  the second deritaives instead of first derivatives, optimization, consider when the training set is sparse. optimize on the memory and computation.

Random forest? (Intuition):

- Underlying principle: several weak learners combined provide a strong learner
- Builds several decision trees on bootstrapped training samples of data
- On each tree, each time a split is considered, a random sample of mm predictors is chosen as split candidates, out of all pp predictors
- Rule of thumb: at each split m=p‾√m=p
- Predictions: at the majority rule
Why is it good?
- Very good performance (decorrelates the features)
- Can model non-linear class boundaries
- Generalization error for free: no cross-validation needed, gives an unbiased estimate of the generalization error as the trees is built
- Generates variable importance
bad：
less powerful in dataset which has a lot noises

\
&nbsp;

## 9. Naive Bayes

---

- based on Bayes theorem
- Assumption: features are independent with each other
- Pros:
  - the results are easy to understand
- Cons:
  - need to calculate prior probaibllity


\
&nbsp;

## 10. K-nearest neighbor

---

- When predicting a new value, estimate based on K points which are mostly closet to this point.
- important parameter:
  - how to choose K
  - how to calculate the distance
- K denotes the k labels that are closest to your target
- The higher K is, the smoother (more robust) model you get
- Several definition for distance: Euclidean, Gaussian, Cityblock
- Implement the k shortest distance from your returned unsorted distance array
- KNN can often product classifiers that are surprisingly close to the optimal Bayes classifer
- As. K grows, the method becomes less flexible and produces a decision boundary close to linear
- pros:
  - easy to understand
  - training speed is fast
  - not that sensitive to abnormal value
- cons:
  - save all the training data

\
&nbsp;

## 11. SVM

---

- A type of binary classification model, maximize the gap on feature space
- support vector: the datapoint in sample which are closest to hyperplane
- pros:对特征空间划分的最优超平面是SVM的目标,最大化分类边际的思想是SVM方法的核心；
- cons: bad performance on big data, have some problems on multi classification

\
&nbsp;

## 12. PCA

---

- By linear transformation, a dimensionality reduction method that maps data to a low-dimensional subspace to prevent information loss
- Statistical method that uses an orthogonal transformation to convert a set of observations of correlated variables into a set of values of linearly uncorrelated variables called principal components.
- PCA is linear combinations of the predictor variables.
-
- Dimensionality reduction
  - Remove feature correlation
  - Reduce model overfitting
  - It captures the most of the ‘variability’ in the original data and minimized the reconstruction error in lower-dimension
  - Using the orthogonal transformation to transform the variables which are correlated with each other to the variables which are not correlated.
- Weights: component loading. These transform the original variables into the principal components.

Reduce the data from nn to kk dimensions: find the kk vectors onto which to project the data so as to minimize the projection error.

- How to compute?
  - in creating the first principle component, PCA arrives at thhe linear combination of predictor variables that maximizes the percent of total variance explained.
  - the linear combination ten becomes the first "new " predictor Z1
  - repeat the process to get z2...Zi
- how many components to choose?
  - use an ad hoc rule to select the components that explain "most" of the variance.
    - draw a screeplot
    - set a threshold: such as over 80% cumulative variance
    - inspect the loadings to determine if the component has an intuitive interpretation
Algorithm:

1) Preprocessing (standardization): PCA is sensitive to the relative scaling of the original variable
2) Compute covariance matrix ΣΣ
3) Compute eigenvectors of ΣΣ
4) Choose kk principal components so as to retain xx% of the variance (typically x=99x=99)
Applications:
1) Compression

- Reduce disk/memory needed to store data
- Speed up learning algorithm. Warning: mapping should be defined only on training set and then applied to test set

2. Visualization: 2 or 3 principal components, so as to summarize data
Limitations:

- PCA is not scale invariant
- The directions with largest variance are assumed to be of most interest
- Only considers orthogonal transformations (rotations) of the original variables
- PCA is only based on the mean vector and covariance matrix. Some distributions (multivariate normal) are characterized by this but some are not
- If the variables are correlated, PCA can achieve dimension reduction. If not, PCA just orders them according to their variances

\
&nbsp;

## 13. Feature Selection

---

- Advantages:
  - Better understanding of your model - interpretation
  - Improve model stability(improve generalization)
    - Reduce overfitting
    - multicollinearity
- Method: The effect of ridge regression and lasso regression
  - Ridge: the coefficient will tend to be the same after regularization, spread out more equally. And it is more stable.
  - Lasso: one of the coefficient tend to be 0 after regularization, the sparse solution
- Pearson Correlation: it is used to measure linear dependency between features.

Cov means covariance between x1,x2, sigma means standard deviation

- Other methods
  - In some non-linear model such as Random forest  ,they already have the attribution of feature importance.

\
&nbsp;

## 14. K-means

---

- K-means algorithm:
  - For k-means multiple times
    - Reset centroids
    - For each initialized centroid, execute steps in feature above
  - Kmeans algorithm cannot make sure to converge very quickly. In worst case, it may not coverge at all. Therefore, we need to run k-means multiple times and set termination condition for each run
  - Calculate satisfy value for each k-means run: measure the cluster diversion of k-means result
- K-means ++
  - Algorithm for choosing the centroids for the k-means clustering algorithm in order to avoid the sometimes poor clustering found by standard k-means
  - Kmeans performs better by spreading out the k-initial cluster centroids
  - Steps
    - Choose one centroid uniformly from the data point
    - For each data point x, compute the distance distance(x) between x and the nearest center that has already been chosen
    - Choose one new data point randomly as a new seed, using a weighted probability distribution where a point x is chosen with probability proportional to distance(x) * distance(x)
    - Repeat step b and step c until we get all the k centroids
    - d

Hierarchical Clustering
---

- accommodates non-numerical variables

---

\
&nbsp;

## 15. How to deal with Missing Value

- Delete
- Imputation
  - Categorical feature:
    - Create a new categorical
      - Logistics regression
  - Continuous feature:
    - Mean, Median,mode
    - Regression
      - May cause multicollinearity

\
&nbsp;

## 15. How to deal with Categorical Variable

---

- One Hot Encoding:
  - The calculation of distance between features is more appropriate
  - Cons: the number of features will be increased.
  - Tree don’t need to do one hot encoding, one hot will increase the depth and tree model only do the selection but not are comparing the feature value
- Label Encoding

\
&nbsp;

## 17. How to deal with imbalance problem

---

- Expand dataset
- Try other evaluation metrics
  - If using logic regression, try confusion matrix, precision, recall, accuracy, Roc
- Resampling
  - Oversampling the small category (get more samples which is more than sample number of small categories sample)
  - Undersample the big category (get more samples which is less than sample number of small categories sample)
  - SMOTE：pick one positive category, find the k nearest sample, and at the  median distance between this point to its neighbor, create a new sample
- Consider other method
  - Decision tree has a better performance on imbalance data

\
&nbsp;

## 18. Deal with Collinearity problem

---

- Increase the sample
- Using PCA
- Lasso Regression
Collinearity/Multicollinearity:
- In multiple regression: when two or more variables are highly correlated
- They provide redundant information
- In case of perfect multicollinearity: β=(XTX)−1XTyβ=(XTX)−1XTy doesn’t exist, the design matrix isn’t invertible
- It doesn’t affect the model as a whole, doesn’t bias results
- The standard errors of the regression coefficients of the affected variables tend to be large
- The test of hypothesis that the coefficient is equal to zero may lead to a failure to reject a false null hypothesis of no effect of the explanatory (Type II error)
- Leads to overfitting
Remove multicollinearity:
- Drop some of affected variables
- Principal component regression: gives uncorrelated predictors
- Combine the affected variables
- Ridge regression
- Partial least square regression
Detection of multicollinearity:
- Large changes in the individual coefficients when a predictor variable is added or deleted
- Insignificant regression coefficients for the affected predictors but a rejection of the joint
hypothesis that those coefficients are all zero (F-test)
- VIF: the ratio of variances of the coefficient when fitting the full model divided by the variance of the coefficient when fitted on its own
- rule of thumb: VIF>5VIF>5 indicates multicollinearity
- Correlation matrix, but correlation is a bivariate relationship whereas multicollinearity is multivariate

collaborative filtering, n-grams, cosine distance?

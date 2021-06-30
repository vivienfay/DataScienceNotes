i.        比较mean (t-test, z-test, mann whitney test, paried t-test, signed rank test), proportion (z-test), 2 by 2 table (chi-sq), 比较distribution (K-S test)；另外非参的还有permutation, bootstrap, jackknife。

ii.        然后就是power, type I error, effective size一套, 比如given power and significance level, 算算 sample size算算effective size， 这一套叫power analysis, sample size calculation。

iii.        接着还有multiple testing 咋处理，比如bonferroni，fdr adjustment

iv.        常见问题还有，plain language 解释p value, 95% confidence interval

        OLS的一整套, assumption，estimation, test (t test, F test) , diagnostic， model selection ：常见的问题除了基础知识也有不少，比如multicollinearity (how to identify it, how to deal with it) ，比如list all assumptions of OLS, write down or derive beta estimates (using max likelihood or least squares) ，比如可以用BIC啥的选subset of features， 比如R squared 的几何解释，比如如果double observation ，prediction和 coef estimation会怎么变，

•        Logistic一整套，log(p/1-p), likelihood等等基础知识，拓展一点的话就是用logistic做classification, 可以涉及比如cut off怎么选( error rate, auc, cross validation) , 比较一下logistic versus LDA

•        稍微高级一点：Ridge vs lasso，啥时候用ridge 啥时候用lasso, 聊聊优缺点，讲讲bias-variance trade-off, 说说你怎么选择tuning parameter lambda

•        PCA, 可以作为解决collinearity 的一种方法，写一下SVD, 说说跟SVD有啥联系，有啥缺点


\
&nbsp;

## 1. Probability Foundation

---

- Probability: P(A) = m/n
- Conditional Probability:
  - P(A|B) = P(A,B)/P(B)            P(A|B) * P(B) = P(A,B)
  - Chain Rules: P(A,B,C) = P(A|B,C) *P(B,C) = P(A|B,C)*P(B|C)*P(C)
  - Probability of lottery: Everyone should be the same

    ```
    Proof:
    N people for 1 lottery
    1st: 1/n
    2nd: (1-1/n) * 1/(n-1) = 1/n
    3rd: (1-1/n) * (1-1/n-1) * 1/(n-2) = 1/n
    ```

 - Bayes’ theorem: P(B|A) = P(B) * P(A|B) / P(A) 
   
 -  bournoulli distribution
  - 
    
  - 
 - Binomial: 
   
 - 
   
 - 
 - Polynomial: 
   
- Random Variable: Discrete / Continuous
  - Discrete
    - Expectation
    
    , p is probability
    - Variance
    
    , describe the discrete degree
    - Discrete Probability Distribution
      - Bernoulli distribution(two point distribution)
     
      - Binomial Distribution
     
     (
  - Continuous
    - Using Probability Density function to describe the probability (Every specific number is 0, meaningful only in an area)
    
    - Cumulative Density Function
    
  - Continuous Probability Distribution
   - Normal Distribution
     
     (remember the formula)

- Large Number Therom:
  - the average of the results obtained from a large number of trials should be close to the expected value and will tend to become closer to the expected value as more trials are performed.
- Central Limit Theorem
  - If the sample size is big enough and they are all independent, this sample is similar with normal distribution (independent and identically distributed random variables)
  - use case:
    - Used in hypothesis testing
    - Used for confidence intervals
    - Random variables must be iid: independent and identically distributed
    - Finite variance

2. Statistics Foundation

---

- Basic Description
  - Numeric
    - Spread: variance, std, min, max, quantile,range
    - Center: mean(more stable, include outlier, follow center theroy)/median
  - Categorical
    - Frequency, percentage, proportion of each category
- Variable
  - Nominal: Categorical variables with no inherent order or ranking sequence
  - Ordinal: with an inherent rank or order
  - Continuous
- Bivariate: correlation / regression
  - Confusion matrix
  - What if discrete data?
  - What if the relationship between continuous data and discrete data
    - boxplot
    - histogram
    - Logit regression
  - Covariance:
   
  - Correlation coefficient:
   
    [-1,1]
  - r^2 = 1- sum((y_pred - y)^2)/ sum((y-y_mean)^2)

- Distribution:
  - Unimodal - single peak
  - Bimodal - two distinct peak
  - Symmetric
  - Using Frequency table / density function / cumulative frequency / cumulative distribution function
  - Discrete Distribution:
    - Bornoulli distribution
      - for single item, the outcome only has two types
    - Categorical distribution
      - For single item, the outcome has more than two types
    - binomial distribution
      - multiple item which are independent with each other,  and each event follows Bernoulli distribution
    - Multinomial distribution
      - multiple item which are independent with each other,  and each event follows categorical distribution
    - Poisson distribution
      - in a period of time, the number of times that occurred
    - Geometric distribution
      - For items which follow normal distribution, achieve the first success until k times trails
  - Sampling Distribution:
    - Normal distribution
   -
    - Standardized normal distribution：
    
      - 68-95-99.7 rules
    - t-distribution(the sample from which follows normal distribution)
      - 对df取极限和正态分布重合（add one more parameter(degree of freedom)than normal distribution）
      -
    - chi-square
      - the sample’s square and samples are from normal distribution
      - 对服从标准正态分布的多个相互独立的变量做平方和
      -
    - F-distribution
      - the ratio of a and b and both of them follow chi-square distribution
      - the ratio of two independent variable which follows the chi-square distribution divided by their corresponding degree of freedom

3. Hypothesis Testing

---

- P-value: under the null hypothesis, the probability of obtaining as or more extreme results than the current observation
- Significance level: the probability of rejecting H0, when H0 is true
- Significance Power: the probability of reject h0 when h1 is true
- confidence level: the percentage of all possible samples that can be expected to include the true population parameter
- Confidence interval: the range within which the mean is expected to fall in multiple trials of the experiments
  
- t-test: test if the difference between mean of two sample is statistically difference so that we are able to say two population is also different.(sample size is small and the variance of sample is unknown)
- Z-test: when population variance is known and the sample size is larger than 30, infer the probability of achieving the difference to compare if the diff between mean of two samples is significant or not
  - can test the difference between ratio of samples
  -
   
  -
- Difference between t-test and z-test: if don’t know variance of population, use t-test, otherwise use z-test. Also when sample size > 30, use z-test.
- margin of error: tells you how many percentage points your results will differ from the real population value.
- Anova Test(F-test):
  - 指的是利用对多个样本的方差的分析，得出总体均值是否相等的判定
  - When two sample t test, test if the variance of two sample is statistically different
  - compare if two or more sample mean has statistically difference(continuous)
- Chi-sqaure test:
  - determine whether there is a statistically significant difference between the expected frequencies and the observed frequencies in one or more categories of a contingency table.
  - se the chi-square goodness of fit test to determine whether observed sample frequencies differ significantly from expected frequencies specified in the null hypothesis.
  - test if two categorical variables have any relationship
  - Test two or more sample ratio has statistically different
  -
- Two-side/one-side:
- H0 is the hypothesis you want to subvert, H1 is the hypothesis you want to prove
  - Is a better than b?  h1:a-b>0
  - Can you tell me if there is a difference? H0: a - b = 0
  - Which one is better? Ask if we can set as two-side
- If we cannot reject H0, we don’t have sufficient evidence to make conclusive statement.
  - Collect more data, run longer experience
  - Improve the metric estimation
  - If only one metric is different
    - Segment Analysis: based on different platform, different time period
    - Cross Validation:
  - If more metrics are different
    - Use the higher confidence level
      - If each metric are independent with each other: make
     
      - Beferroni correction
- Error
  - Type 1 Error: This happens when H0 is true but is rejected, significance level = type I error rate = probability of type I error
  - Type 2 Error(beta): This happens when H1 is true but h0 is not rejected, type 2 error rate = 1- probability of type I error
  - Power: the probability of rejecting H0 when h1 is true. Power = 1- beta
  - Tradeoff: convicting an innocent man(Type i ) vs realeasing a criminal (type ii). Type I error is associated with a larger risk than type ii error is.
- befferroni correction:
- Hypothesis Testing Formula (Applicable for mean/average: CTR /DAU/ ARPU)
  - scipy.stats.ttest_ind

  - If the variable is neither binary nor continuous, such as multi-level categorical, make it as binary data

4. Simpson paradox

---

- Simpson paradox
            Though the trends appear in all groups, after we combine the groups the trends are disapear
- How to determine if it is independent? Union probability

- Check any confounders:
  - The results are replicated in many studies
  - Each of the studies controlled for plausible confounding variables
  - There is a plausible scientific explanation for the existence of a casual relationship

- How to estimate the strength of dependence
 -
  - Two continuous variables: look at their correlation, if and only if two variables are normally distributed
  - One continuous variables and one discrete: make KS test

5. linear Regression

---

- Assumption:
  - 自变量（X）和因变量（y）线性相关
  - 自变量（X）之间相互独立
  - 误差项（ε）之间相互独立
  - 误差项（ε）呈正态分布，期望为0，方差为定值
  - 自变量（X）和误差项（ε）之间相互独立
- ols assumptionn：
  - independent variable are normally distributed  in terms of specific dependent variable value
  - independent
  - independent variables and dependent variables are linear correlatetd
  - variance of dependent variable are constant
- BIC的惩罚项比AIC的大，考虑了样本数量，样本数量过多时，可有效防止模型精度过高造成的模型复杂度过高。

6. Time Series

---

- Stationary: expectation, variance = constant, covariance(zt, zt+k) depends on k, is constance
- White Noise: expectation = 0, variance = constant, Covariance(zt, zt+k) = 0. Ljung-Box test(ARIMA residuals)
- General approach to time series Modeling
  - Exploratory analysis (Plot)
    - Trend(MK test)
    - Seasonal component
    - Apparent sharp changes in behavior
    - Outlying observation
  - Check if it is stationary
    - ADF test
    - KPSS test
  - Get the stationary residuals
    - Remove trend and seasonal component
      - large fluctuation (large variance):  log transformation
      - trend:  first order differencing
      - seasonal: s order differencing
    - Check if it’s stationary again using tests above
  - Choose a model to fit the residuals to make use of various sample statistics including the sample autocorrelation function
    - How to choose the model
    - How to estimate the model parameters
  - Forecasting will be achieved by forecasting the residuals and then inverting the transformation
    - Why to forecast the residuals

- Weakly Stationary
 -

- Model:
  -
  - ARIMA model(p, q):
    - Component:
      - AutoRegressive: order p
      - Integrated Differencing: d
      - Moving Average: order q
    - Decide p, d, q:
      - ACF: auto correlation function
     
      - Partial ACF:
      - ACVF: auto covariance function
     
    - Estimate the parameter: OLS
- Evaluation:
  - Mean error:
  - Mean absolute deviation:
  - Mean square error

---
Common tests:

- One sample Z test

- Two-sample Z test

- One sample t-test

- paired t-test

- Two sample pooled equal variances t-test

- Two sample unpooled unequal variances t-test and unequal sample sizes (Welch’s t-test)

- Chi-squared test for variances

- Chi-squared test for goodness of fit

- Anova (for instance: are the two regression models equals? F-test)

- Regression F-test (i.e: is at least one of the predictor useful in predicting the response?)

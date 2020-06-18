# Machine Learning (MATLAB) - Regularized Linear Regression and Bias/Variance

Machine Learning course from Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/programming/Im1UC/regularized-linear-regression-and-bias-variance).

### Environment
- macOS Catalina (version 10.15.3)
- MATLAB 2018 b

### Dataset
- ex5data1.mat

### Files included in this repo
- ex5.m - Octave/MATLAB script that steps through the exercise 
- ex5data1.mat - Dataset
- submit.m - Submission script that sends solutions to our servers 
- featureNormalize.m - Feature normalization function
- fmincg.m - Function minimization routine (similar to fminunc) 
- plotFit.m - Plot a polynomial fit
- trainLinearReg.m - Trains linear regression using cost function 

[⋆] linearRegCostFunction.m - Regularized linear regression cost function

[⋆] learningCurve.m - Generates a learning curve

[⋆] polyFeatures.m - Maps data into polynomial feature space 

[⋆] validationCurve.m - Generates a cross validation curve

### Part 1: Regularized Linear Regression
We will implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir.

#### Part 1.1: Loading and Visualizing Data  
We will begin by visualizing the dataset containing historical records on the change in the water level, x, and the amount of water flowing out of the dam, y.

This dataset is divided into three parts: 
- A training set that your model will learn on: X, y  
- A cross validation set for determining the regularization parameter: Xval, yval 
- A test set for evaluating performance. These are “unseen” examples which your model did not see during training: Xtest, ytest  

```
% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');
```

![plot](Figure/dataset.svg)
- Figure: Dataset


#### Part 1.2: Regularized Linear Regression Cost  
The regularized cost function for linear regression:  
![plot](Figure/cost.png)  
where λ is a regularization parameter which controls the degree of regularization (thus, help preventing overfitting). The regularization term puts a penalty on the overal cost J. As the magnitudes of the model parameters θj increase, the penalty increases as well.


### Part 4: Train Linear Regression 


### Part 5: Learning Curve for Linear Regression  


### Part 6: Feature Mapping for Polynomial Regression 


### Part 7: Learning Curve for Polynomial Regression  


### Part 8: Validation for Selecting Lambda 


## Course Links 

1) Machine Learning by Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/programming/Im1UC/regularized-linear-regression-and-bias-variance).

2) [Regularized Linear Regression](https://www.coursera.org/learn/machine-learning/home/week/6)
(Please notice that you need to log in to see the programming assignment.) #ML-Neural_Networks_Learning-MATLAB
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
- sigmoid.m - Sigmoid function
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


#### Part 1.2: Regularized Linear Regression Cost Function  
The regularized cost function for linear regression:  
![plot](Figure/cost.png)  
where λ is a regularization parameter which controls the degree of regularization (thus, help preventing overfitting). The regularization term puts a penalty on the overal cost J. As the magnitudes of the model parameters θj increase, the penalty increases as well.

##### linearRegCostFunction.m 
```
% Regularized linear regression cost function
h = (X * theta);
J = (1/(2*m)) * sum((h-y).^2) + (lambda/(2 * m)) * sum(theta(2:end).^2);
```

Result: Cost at theta = [1 ; 1]: 303.993192 

#### Part 1.3: Regularized linear regression gradient   
The partial derivative of regularized linear regression’s cost for θj is defined as
![plot](Figure/gradient.png)  

##### linearRegCostFunction.m 
```
% Regularized linear regression gradient function
grad = (1/m * X' * (h - y)) + [0;(lambda/m) * theta(2:end)];
```

Result: Gradient at theta = [1 ; 1]:  [-15.303016; 598.250744] 


#### Part 1.4: Fitting Linear Regression 
Once we get done with implementing cost and gradient function, we need to compute the optimal values of θ. *trainLinearReg.m*, this training function uses fmincg to optimize the cost function.

##### trainLinearReg.m - Trains linear regression using cost function 
```
% Initialize Theta
initial_theta = zeros(size(X, 2), 1); 

% Create "short hand" for the cost function to be minimized
costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');

% Minimize using fmincg
theta = fmincg(costFunction, initial_theta, options);
```

We set regularization parameter λ to zero. Because our current implementation of linear regression is trying to fit a 2-dimensional θ, regularization will not be incredibly helpful for a θ of such low dimension.

The best fit line tells us that the model is not a good fit to the data because the data has a non-linear pattern.

![plot](Figure/linearfit.jpg) 
- Fig. Linear Fit

In the next section, we will implement a function to generate learning curves that can help debug our learning algorithm even if it is not easy to visualize the data.


## Part 2: Bias-variance
An important concept in machine learning is the bias-variance tradeoff. Models with high bias are not complex enough for the data and tend to underfit, while models with high variance overfit to the training data.

We will plot training and test errors on a learning curve to diagnose bias-variance problems.

### Part 2.1: Learning Curve for Linear Regression  
We will now implement code to generate the learning curves that will be useful in debugging learning algorithms. We fill in learningCurve.m so that it returns a vector of errors for the training set and cross validation set.

To plot the learning curve, we need a training and cross validation set error for different training set sizes. To obtain different training set sizes, we should use different subsets of the original training set X. Specifically, for a training set size of i, we should use the first i examples (i.e., X(1:i,:) and y(1:i)).

We use trainLinearReg function to find the θ parameters. Note that the lambda is passed as a parameter to the learningCurve function. After learning the θ parameters, we should compute the error on the train- ing and cross validation sets. 
The training error for a dataset is defined as
![error](Figure/error.png)  

The training error does not include the regularization term. One way to compute the training error is to use your existing cost function and set λ to 0 only when using it to compute the training error and cross validation error.

When you are computing the training set error, make sure we compute it on the training subset (i.e., X(1:n,:) and y(1:n)) (instead of the entire training set). However, for the cross validation error, we should compute it over the entire cross validation set. we should store the computed errors in the vectors error train and error val.

##### learningCurve.m
``` 
%Generates a learning curve

```

![learningcurve](Figure/curve.jpg)
- Figure: Linear Regression learning curve  

We can observe that both the train error and cross validation error are high when the number of training examples is increased. This reflects a high bias problem in the model – the linear regression model is too simple and is unable to fit our dataset well.



### Part 2.2: Feature Mapping for Polynomial Regression 


### Part 7: Learning Curve for Polynomial Regression  


### Part 8: Validation for Selecting Lambda 


## Course Links 

1) Machine Learning by Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/programming/Im1UC/regularized-linear-regression-and-bias-variance).

2) [Regularized Linear Regression](https://www.coursera.org/learn/machine-learning/home/week/6)
(Please notice that you need to log in to see the programming assignment.) #ML-Neural_Networks_Learning-MATLAB
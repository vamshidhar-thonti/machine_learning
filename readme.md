# Machine Learning

> **Definition:** Field of study that gives computers the ability to learn without being explicitly programmed. _- Arthur Samuel_

- Machine Learning is classified into 2 types:
  1. **Supervisied Learning:** When an input of x is given and an output of y is expected. We train the model with some known information and then give a completely unknown input to the trained model to predict the output based trained data. The data is always in a structured format.
  2. **Unsupervisied Learning:** Unlike supervised learning, here the outcome will not be predicted or classified rather with the input data, information is obtained. Clustering is one such algorithm under it.

## Supervisied Learning

2 categories of it are _Regression_, _Classification_

> **Regression** is when a input is given, the model predicts a single output from an infinite number of numbers.

> **Classification** is when an input is given, the model classifies it into the predefined categories/classes. (Predicts from a small number of categories)

## Unsupervisied Learning

> **Clustering** is when the input data is grouped together to give sufficient information. For example, grouping of news or providing suggestions based on the viewed videos in youtube etc.

## Terminology

The linear regression model's function is basically defined with the below function:
$$f_w,_b\left(x^{(i)}\right)=wx^{(i)}+b$$
Where

- w, b are called `parameters`, `coefficients`, `weights`
- x is `feature`
- y is `target`
- $x^{(i)}, \space y^{(i)}$ is $i_{th}$ Training example
- m is Number of training examples
- $\^y^{(i)}$ is estimated output for a given model

The estimated/predicted output function can be defined as below:
$$\^y^{(i)} = f_{w,b}(x^{(i)})$$
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b$$

### Cost function

This functions determines the value of deviation from the expected output. The function definition is as below:
$$J\left(w, \space b\right) = \frac{1}{2m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})^2$$
$$or$$
$$J\left(w, \space b\right) = \frac{1}{2m}\sum_{i=0}^{m-1}(\^y^{(i)} - y^{(i)})^2$$
$$Where \space \^y^{(i)} - y^{(i)} \space is \space error$$

So, when the evaluated value of $J \space (Cost \space Function)$ is near to $0$, then the model for that training data suits best.

## Gradient Descent

It may not be always possible to evaluate the cost function with trial and error method to find the minumum value, or with contour graphs to find the best suitable values for the model's parameters. Manually finding out the values become impossible when the parameters count increases.
To make this process easy and find the minimum cost function value, the `gradient descent` algorithm best suits. It is applied for some of the advanced ML/DL models.

Using the formula below, the parameter values should be evaluated **simultaneously** and the best value can be obtained

$$
w = w - \alpha \frac{\partial}{\partial w}J(w,b)
$$

$$
b = b - \alpha \frac{\partial}{\partial b}J(w,b)
$$

$$Where$$
$$\space \space \alpha \space is \space Learning \space rate \space (Positive \space Number \space between \space 0 \space and \space 1)$$
$$\frac{\partial}{\partial b}J(w,b) \space gives \space the \space slope$$

By substituting all the values into the gradient descent formula and further evaluating the expression we end up having the gradient descent formula for linear regression as below:

$$
w = w - \alpha \frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}
$$

$$
b = b - \alpha \frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})
$$

 - For each update step (iteration), the descent path/length becomes smaller and smaller, even with a fixed learning rate ($\alpha$)
 - If the slope (derviate part) is -ve, then the gradient descent diverges and may not reach the minumum at all.
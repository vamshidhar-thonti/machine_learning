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

---

## Linear Regression

The linear regression model's function is basically defined with the below function:

$$f_w,_b\left(x^{(i)}\right)=wx^{(i)}+b$$

Where

- w, b are called `parameters`, `coefficients`, `weights`
- x is `feature`
- y is `target`
- $x^{(i)}, \space y^{(i)}$ is $i_{th}$ Training example
- m is Number of training examples
- $\hat{y}^{(i)}$ is estimated output for a given model

The estimated/predicted output function can be defined as below:

$$\hat{y}^{(i)} = f_{w,b}(x^{(i)})$$

$$f_{w,b}(x^{(i)}) = wx^{(i)} + b$$

### Cost function

This functions determines the value of deviation from the expected output. The function definition is as below:

$$J\left(w, \space b\right) = \frac{1}{2m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})^2$$

$$or$$

$$J\left(w, \space b\right) = \frac{1}{2m}\sum_{i=0}^{m-1}(\hat{y}^{(i)} - y^{(i)})^2$$
$$Where \space \hat{y}^{(i)} - y^{(i)} \space is \space error$$

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

Where

- $\alpha$ is Learning rate (Positive Number between 0 and 1)
- $\frac{\partial}{\partial b}J(w,b) \space gives \space the \space slope$

By substituting all the values into the gradient descent formula and further evaluating the expression we end up having the gradient descent formula for linear regression as below:

$$
w = w - \alpha \frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}
$$

$$
b = b - \alpha \frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})
$$

- For each update step (iteration), the descent path/length becomes smaller and smaller, even with a fixed learning rate ($\alpha$)
- If the slope (derviate part) is -ve, then the gradient descent diverges and may not reach the minumum at all.

---

Most of the advanced models will have more than one `feature`, thus will have more than one parameter. Below is a simple example on how a model with multiple features look like

$$
f(x) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b
$$

$$or$$

$$
f_{\vec{w},b}(\vec{x}) = \vec{w}\cdot\vec{x} + b
$$

```python
# Python's implementation of vectorization of the above formula
# This numpy way of implementation is faster than the for loop implmentation
f = np.dot(w, x) + b
```

## Feature Scaling

When the feature values very large or very small, the prediction may show incorrect data, so, it is recommended to scale the value accordingly.
It is suggested to have the values in the following ranges
`-1 to 1`
`-3 to 3`
`-0.3 to 0.3`
If the feature values goes beyond the above ranges, data has to be rescaled.

It can be done in 3 ways:

1.  **Dividing all the features values with the greatest of all**, this leads to the feature rescaling to fall between 0 and 1
2.  **Mean Normalization**, mean of the data is used to rescale
3.  **Z-score Normalization**, standard deviation is used to rescale the data.

## Checking gradient descent for Convergence

`Convergence` happens when the gradient descent is at the minumum value, in another way it can be said that the parameters are close to the global minimum.
\
To make sure if gradient descent is converging, it is best to plot a graph between the iterations and the cost, after the first iteration if the graph is decreasing, we can be sure that the gradient descent is converging, else the graph is said to be diverging (where the gradient descent never reach the minimum) which means `the chosen learning rate is too large` or `there may be a bug in the code`.
\
\
**But how to decide if the gradient descent has reached global minumum (where the parameters are optimal for the training set)?**
\
As the number of iterations to reach the convergence varies for each model, its hard to depend on number of iterations.
\
There are 2 ways to determine it though:

1. Simply looking at the graph and see if the cost remains constant.
2. `Automatic Convergence Test`, when the cost decreases by $\leq\epsilon$ $(10^{-3})$ (epsilon) in one iteration, then declare convergence.

## Choosing the learning rate

As seen before, $\alpha$ varies between 0 and 1. So, choosing a correcting learning rate plays vital role in convergence.
\
It's best to start $\alpha$ with 0.001 and going 3X from there. Plotting the graph with those ranges the learning rate $(\alpha)$ can be determined.

Just remember that choosing a very small $(\alpha)$ may decreases the cost very slow and choosing a very large $(\alpha)$ may lead to divergence.

## Feature Engineering

It's an idea to either transform a feature or combine more than feature to create a brand feature which leads to better prediction from the model.
\
For example, with a model that can predict house prices with length and width as features, we can further engineer the existing features to create a brand new feature called area that helps in predicting the price of a house even better.
\
The more reasonable features in a model, the better prediction can be.

## Polynomial Regression

Often times the training data can be plotted as a curve and the linear regression with straight line will not fit the data. So solve this we can modify the model to have quadratic equations with features being raised to a certain power (say power of 2, power of 3 so on)

> _Its recommended not to end the model's equation with power of 2 because the curve of the model may go down after a certain number of iterations. So, have the quadratic equation for a given feature end with atleast power of 3._

## Classification with linear regression model

Linear Regression models best suits to predict over a range of numbers but in classification, the numbers to be predicted will be handful. Even though this model works better to classify with minimal data points by assigning a threshold (descision boundary), it starts to wrongly classify the data when the model is trained with data points that are spread across a wide range.

> **Linear Regression model is NOT ideal for classification**

Thus, a need of new model arises to solve classfication related problems.

## Logistic Regression

As the motto of this algorithm is to classify the values with either $0$ or $1$. The `sigmoid function` can be better leveraged. Using this function, any output can be mapped to be a value between $0$ and $1$, and by adding a `descision boundary`, the prediction can be classified better.

The formula of `sigmoid function` is as below:

$$
g(z) = \frac{1}{1+e^{-z}}
$$

Where $z$ is the resultant of a linear regression model

$$
z=\vec w \cdot \vec x + b
$$

$$
f_{\vec w,b}(\vec x)=g(\vec w \cdot \vec x + b)
$$

Hence,

$$
g(z) = \frac{1}{1+e^{-(\vec w \cdot \vec x + b)}}
$$

which results the output between the range of 0 and 1.

## Cost function for Logistic Regression

The mean squared error cost function that has been used for linear regression causes a wavy convex function many possible local minimas which is not ideal to find cost for Logistic regression. So, a new formula should be used to calculate cost of the latter model.
In detail,

- when the model predicts ~1 and target is also 1 the loss is said to be minimum.
- when the model predicts ~0 and target is 1 the loss is said to be maximum, which then means the parameters has to be modified.

Similar scenario applies when the target is 0.

The cost function can be calculated as below:

$$
J(w, b) = \frac{1}{m} \sum_{i=0}^{m-1}[loss(f_{w,b}(x^{(i)}), y^{(i)})]
$$

Further the loss function can be simplified as,

$$
loss(f_{w,b}(x^{(i)}), y^{(i)}) = -y^{(i)}log(f_{w,b}(x^{(i)})) - (1 - y^{(i)})log(1 - f_{w,b}(x^{(i)}))
$$

When

- $y = 0$

$$
loss(f_{w,b}(x^{(i)}), y^{(i)}) = -log(1 - f_{w,b}(x^{(i)}))
$$

- $y = 1$

$$
loss(f_{w,b}(x^{(i)}), y^{(i)}) = -log(f_{w,b}(x^{(i)}))
$$

> _**Reminder:** The lower the cost value, the better the model can predict._

Therefore, the cost function can be written as

$$
J(w, b) = \frac{1}{m} \sum_{i=0}^{m-1}[-y^{(i)}log(f_{w,b}(x^{(i)})) - (1 - y^{(i)})log(1 - f_{w,b}(x^{(i)}))]
$$

$$
or
$$

$$
J(w, b) = -\frac{1}{m} \sum_{i=0}^{m-1}[y^{(i)}log(f_{w,b}(x^{(i)})) + (1 - y^{(i)})log(1 - f_{w,b}(x^{(i)}))]
$$

## Gradient Descent for Logistic regression

This is similar to the one that we have for Linear regression

$$
w_j = w_j-\alpha \frac{\partial}{\partial{w_j}}J(w_j,b)
$$

$$
b = b-\alpha \frac{\partial}{\partial{b}}J(w_j,b)
$$

Where

$$
\frac{\partial}{\partial{w_j}}J(w_j,b) = \frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)})-y^{(i)})x_j^{(i)}
$$

$$
\frac{\partial}{\partial{b}}J(w_j,b) = \frac{1}{m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)})-y^{(i)})
$$

Therefore,

$$
w_j = w_j-\alpha \left[\frac{1}{m}\sum_{i=0}^{m-1}(f_{\vec w,b}(\vec x^{(i)})-y^{(i)})x_j^{(i)}\right]
$$

$$
b = b-\alpha \left[\frac{1}{m}\sum_{i=0}^{m-1}(f_{\vec w,b}(\vec x^{(i)})-y^{(i)})\right]
$$

Where

$$
f_{\vec w,b}(\vec x)=\frac{1}{1+e^{-(\vec w \cdot \vec x + b)}}
$$

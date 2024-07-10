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
$$J\left(w, \space b\right) = \frac{1}{2m}\sum_{i=1}^m(f_{w,b}(x^{(i)}) - y^{(i)})^2$$
$$or$$
$$J\left(w, \space b\right) = \frac{1}{2m}\sum_{i=1}^m(\^y^{(i)} - y^{(i)})^2$$
$$Where \space \^y^{(i)} - y^{(i)} \space is \space error$$

So, the when evaluated value of $J \space (Cost \space Function)$ is near to $0$, then the model for that training data suits best.
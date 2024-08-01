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

---

As seen till now, we create a model that suits the training data well but when the features are less in number then the model will not be trained well enough which causes underfit, and when the features are many in number, the model may seem to perfectly fit the training data but the model cannot be generalized as the descision boundary is very complex. The parameters have to be chosen just right so that the decision boundary can be generalized.

## Underfitting (high bias)

This happens when the model's curve is no way near the training set.

## Generalization [recommended]

The model with parameters which is just right, so that the prediction on new examples out of the training set is almost near to the expectations.

## Overfitting (high variance)

This happens when the model's curve exactly fits the training set but practically it may predict wrong.

- **Solution:**
  1. Add more training data
  2. Remove some of the features by making its parameter zero
  3. Make parameter ($w_j$) value low but not $b$. This method is called **_Regularization_** (recommended).

## Cost function with regularization

$$
J\left(w, \space b\right) = \frac{1}{2m}\sum_{i=1}^{m}(f_{\vec w,b}(\vec x^{(i)}) - y^{(i)})^2 + \frac {\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

Where

- $\lambda$ is the _*Regularization parameter*_
  - When its higher the curve remains almost constant with a straight line
  - When its smaller the curve overfits
  - Better to choose a value in between
- $m$ is the number of training examples
- n is the number of parameters

## Gradient Descent with regularization

The updated gradient descent evaluated below is same for both `Linear Regression` and `Logistic Regression` models. Just that the $f(x)$ will differ.

$$
w = w - \alpha \frac{\partial}{\partial w}J(w,b)
$$

$$
b = b - \alpha \frac{\partial}{\partial b}J(w,b)
$$

But now as $J(w,b)$ has been modified, below would be the new expression

$$
w_j = w_j - \alpha \left[ \frac{1}{m}\sum_{i=1}^{m} \left[(f_{w,b}(x^{(i)}) - y^{(i)})x_j^{(i)}\right]+\frac{\lambda}{m}w_j\right]
$$

$$
w_j = w_j - \alpha \frac{\lambda}{m}w_j - \alpha \frac{1}{m}\sum_{i=1}^{m} \left[(f_{w,b}(x^{(i)}) - y^{(i)})x_j^{(i)}\right]
$$

Therefore,

$$
w_j = w_j \left(1 -  \alpha \frac{\lambda}{m}\right) - \alpha \frac{1}{m}\sum_{i=1}^{m} \left[(f_{w,b}(x^{(i)}) - y^{(i)})x_j^{(i)}\right]
$$

$$
b = b - \alpha \frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})
\tag {remains same}
$$

# Neural Networks

Neural Networks uses the same old models but are arranged in a manner that makes the prediction even better.

The basic layers of a Neural network are

- **Input Layer** which contains features represented as a vector.
- **Hidden Layer** which contains more than one model (called as a neuron) that takes all the features as input and return a value called `activations` which are then fed to the next layer. (The next layer can be an another hidden layer or a final output layer, it depends on the architecture). Choosing the count of models (neurons) to be in the hidden layer is the architectural decision.
- **Output Layer** which takes `activations` vector as an input and outputs a final prediction.

## Advice for applying Machine Learning

The ML libraries have most of the functionality in built and there is no need to write the code from scratch.

In neural network models with libraries, most of the implementation is done within, like cost, loss calculations and the gradient descent ti minimize the cost etc...

Till now the activation function that we have learnt and most widely used are

1. `Linear activation function (No activation function)` for linear regression.
   $$g(z)=z$$
2. `Sigmoid activation function` for Binary calssification.
   $$g(z) = \frac {1}{1 + e^{(-z)}}$$
3. `Recitified Linear activation function (relu)` for non negative number prediction (House prices).
   $$g(z)=max(0, z)$$

## Choosing Activation funcitons

For the output layer, $g(z)$, the activation function can be chosen based on the output format

If $y$ (output) is

- $0 / 1$, then choose `Sigmoid`
- $+ / -$, then choose `Linear`
- $>=0$, then choose `ReLU`

For hidden layers, its always recommended to use ReLU activation function due its fastness. Because all it does is choose the mac value between $0$ and $z$ that leads to a faster learning process compared to the linear activation function.

### Why do we need activation functions?

It seems that using a linear activation function across all layers leads to a resultant linear activation function which anyhow does not fit all kinds of training data. So, its always recommented not to use linear activaiton function.
\
It is recommended to use a ReLU activation function.

## Multiclass classification

Till now we only saw the binary classification (either 0 or 1) but what if we want classification of more than 2 classes/categories? That is when the Multiclass classification algorithms pitch in.
\
The commonly used multiclass classification algorithm is `softmax`.

### Softmax classification

It is a generalized function of the logistic regression.
\
The formula for the softmax function is

$$
a_1 (output\ of\ class\ 1) = \frac {e^{z_1}} {e^{z_1} + e^{z_2} + ... +e^{z_n}}
$$

$$
a_2 (output\ of\ class\ 2) = \frac {e^{z_2}} {e^{z_1} + e^{z_2} + ... +e^{z_n}}
$$

$$.$$
$$.$$
$$.$$

$$
a_n (output\ of\ class\ n) = \frac {e^{z_n}} {e^{z_1} + e^{z_2} + ... +e^{z_n}}
$$

Where,

$z=w \cdot x + b$
\
and $sum(a_1, a_2, ..., a_n)=1$

Note that the output layer of the neural network contains $n$ neurons instead of 1. Where $n$ is the number of classes/categories.

Numerical round occurs when the implementation does vary slightly and the corresponding output is almost equivalen but not truely equivalent.
\
For example, $\frac {2}{10000}$ and $\left(\left(1 + \frac {1}{10000}\right)-\left(1-\frac {1}{10000}\right)\right)$ may produce almost same result but with minor difference in the decimal points. That difference in decimals is called as `Numerical Round Off Error`, which might seem negligible but impacts lot in case of softmax activation function. Also, using accurate values can lead the model to be more stable.
\
This can be acheived by calculating the loss with the activation functions equations itself rather than evaluatingusing the output of the activation function.

Even in the code implementation we can achieve this a s following (recommended implementation)\
On the output layerm instead of using sigmoid, use linear activation function\
and while compiling, instead of using loss function alone, pass named parameter `from_logits=True`, where logit means to evaluate the loss function with the equation rather than the output of the equation. Thus the model is more numerically accurate.

## Multi-label classification

It may seem similar to that of the multi class classification but its not. Here, we tend to label multiple expectations from a single input rather a single expectation.

For example: detecting car, bus, pedestrian from a single image is considered to a be multi-label classification.\
Where while with handwritten digit classification we get on image with a number and we classify only one value as an output.

## Adam Optimization algorithm:

While gradient descent helps in finding the parameters with low cost/loss, the operations performed to get there is tiny many steps. To make the learning process faster `Adaptive moment estimation` algorithm can be used. Ir can adjust the learning rate accordingly.\
If the steps are too small and yet reached the minimum, then the learning rate will be increased by the algorithm automatically and vice versa.\
One point to note is, the learning rate alpha is not global rather unique to each of the parameter including the parameter $b$.\
So, if there are parameters from $w_1$ to $w_n$ and $b$, then there would be learning rates from $\alpha _1$ to $\alpha _n$ and $\alpha _b$ respectively. Any default learning rate can be assigned initially while using the algorithm.

## Convolutional Neural Network

In a dense layer that we have been using till now actually it means that the layer's unit/neuron takes inputs/activation (if its a layer) as everything but with convolutional neural network or layer, the input of the unit/neuron would be only a part of it unlike with the dense layer. This way the output would be faster compared to the dense layer.

Data should be split into 3 parts

- Training set - 60%
- Cross validation set - 20%
- Test set - 20%

## Bias and Variance

High Bias occurs when the model is underfit.
High Variance occurs when the model is overfit.

Based on the evaluation of cost function of various split data, the respective cost function comparision helps us find out if the model is `high bias`, `high variance` or in some rare scenarios its possible to have both high bias and high variance

## Descision Trees

Decision trees is another algorithm where the inputs are filtered based on the features. It resembles similar to that of a tree structure in DSA. Where top node is called `root node`, all the intermmediate nodes are called `decision nodes` and the last nodes in the tree are called `leaf nodes`.
\
At the decision nodes, the decision of spliting of the inputs are made based on feature that has been used.
\
**Purity** of the decision tree is measured based on the resultant splits that are being made.
\
For example, with a dataset of 20 images consisting of 10 cats and 10 dogs, with features like ears being `pointy` or `floppy`, whiskers being `present` or `absent`, face being `round` or `not round`. With these features being in the decision node the images would split and move down the tree until all the images are classified as either `cat` or `not a cat`.
\
The purity of the algorithm is measured based on the final classfication. If the majority of the inputs are correctly classified as cats by the algorithm, it is said to be having a higher purity and vice versa.

## Decision tree training

### Entrophy as a measure of purity

Entrophy's graph is of bell curve, at extremes of $p$ the Entrophy is less and at the center values of $p$ the Entrophy is high.
\

> Lower the Entrophy, higher the purity and vice versa.

$p1$ is the fraction of examples that are cats
\
For example, if 3/6 are cats then $p1$ is 0.5 and $H(p1)$ is 1

When formulated, Entrophy is given as

$$
H(p1) = -p_1  log_2(p_1) - p_0  log_2(p_0)
$$

$$
p_0 = 1 - p_1
$$

Hence,

$$
H(p_1) = -p_1log_2(p_1) - (1-p_0)log_2(1-p_0)
$$

Note: $$"0log(0)"=0$$

### Choosing a split: Information Gain

Given to classify cats from 10 images, which actually consists of 5 cats and 5 dogs with 5 pointy ear shape and 5 floppy ear shape.

Information gain with the feature ear shape can be given as

$$
= H(p_1^{root}) - (w^{left}*H(p_1^{left}+w^{right}*H(p_1^{right})))
$$

Where,

$p_1^{root}$ is number of images are actually cats = 5/10 = 0.5

$p_1^{left}$ is number of images are actually cats with ear shape as pointy = 4/5

$w^{left}$ is number of images are classified with ear shape as pointy = 5/10

$p_1^{right}$ is number of images are actually cats with ear shape as floppy = 1/5

$w^{left}$ is number of images are classified with ear shape as floppy = 5/10

> Choose the feature which has high information gain to start splitting.

### One hot encoding

Till now, the example had feature like round-not_round, present-absent etc, what if the feature can be classified with more than 2 outputs like ear shape can be pointy, floppy and oval?
/
That is where the one hot encoding helps us

> If a categorical feature can take on $k$ values, create k binary features (0 or 1 valued)

If a feature has continuous value (say weight can any number which means continuous), then take each weight value as threshold and calculate the information gain, the threshold with high information gain can be chosen as the threshold for that decision node.

## Tree ensemble

Instead using just a single decision tree, we can group more than one decision tree which makes the algorithm more robust and efficient, this method is called as tree ensemble.

## Random forest algorithm

At each node, when choosing a feature to use to split, if $n$ features are available, pick a random subset of $k \lt n$ features and allow the algorithm to only choose from that subset of features.

$$k=\sqrt n$$

## Boosted trees:

Given a training set of size $m$

For b = 1 to B:

- Use sampling with replacement to create a new training set of size m
  - But instead of picking from all examples with equal $(1/m)$ probability, make it more likely to pick misclassified examples from previously trained tree.
- Train a decision tree on a new dataset.

In simple words, after iteration of evaluating a model, consider the misclassfied examples with high priority in the following iterations and repeat the same process.

## XGBoost

eXtreme Gradient Boost tree is implemented with the same logic as above but more robust and efficient than the general boosted trees which can regularize the data from overfitting.

## Unsupervised Learning

The main objective here is to solve a problem when the data is not labeled or in other words the data's output is not defined before hand. Typically used to cluster the data, convert the data from one form to another etc...

### K-means

- Helpful in solving the problem where clustering is needed.
- All it does is 2 steps:

  - Given a dataset of 30 examples scattered over the graph into 2 sections/groups. The 2 k-means cluster centroids are randomly placed on the graph at distant places.

    1. The first step of the algorithm is, the nearest examples to the respective centroids are classified and are formed into group.
    2. The second step is, based on the classified group, the mean of the group is calculated and that centroid is moved to the position corresponding to the mean value.

    - The process repeats from step 1 again until there is no further change in the mean value.

- The cost function can either decrease or increase for every iteration but only choose the lowest one.
- The elbow method can be used to choose the number of clusters but its not an optimal method to follow.
- The random initialization of centroids has to be repetitive till we find the centroid locations that has the minimum cost function.

Refer the coursera videos for detailed formulae

### Anomaly Detection:

- Helpful is finding the anomalies in a system given data.
- Can detect the anomalies that haven't occured previously.
- Uses Gaussian (Normal) Distribution to train the model and detect the anomalies.
- Lower the probability likely the anomaly.

Refer the coursera videos for detailed formulae

### Collabarative Filtering

> _Refer coursera videos_

### Content based Filtering (Recommender Systems)

- Given vectors of user information like age, gender, movies watched etc and movies that are being already rated with genre information, this algorithm helps in filtering the content and recommend a similar movie.
- To acheive this the vectors of the user information and the vector of the movies and it's ratings should of equal length.
- To acheive it, we pass the features through a neural network.
  - Pass user information into a neural network with hidden layers of 258, 128, with 32 outputs
  - Pass movie ratings and genres into a neural network with hidden layers of 258, 128, with 32 outputs
- Applying a dot product on both the vectors give a predicting rating.
- With that we can have then find similar movie with distance formula, (which generates a value, smaller the value similar it is). This way recommendation can be made.
  - A similarity measure is the squared distance between the two vectors $\mathbf{v_m^{(k)}}$ and $\mathbf{v_m^{(i)}}$ _(movie vectors)_:

$$\left\Vert \mathbf{v_m^{(k)}} - \mathbf{v_m^{(i)}}  \right\Vert^2 = \sum_{l=1}^{n}(v_{m_l}^{(k)} - v_{m_l}^{(i)})^2\tag{1}$$

### Principal Component Analysis (PCA)

- To visualize a dataset with many features (say 100s), it is impossible to draw a graph and get some insights.
- Using the PCA algorithm we can fortunately visualize it by reducing the features to some handful numbers.
- In the algorithm, with the available data, we choose new axis(s) such a way that the distance from the new axis is reduced and the chosen axis should be able to plot the data divergely but not squished. This way the information can be retained with fewer features that can be easily plotted.
- To reconstruct the original value (appromates) just multiply the new feature value with the vector of distances from the origin of new axis.
- > Refer the coursera videos for detailed information.

# aRUB - PyTorch Loss Function
An Approximate Robust Upper Bound (aRUB) loss function implementation in PyTorch as taken from [Bertsimas et al. (2021)](https://arxiv.org/abs/2112.09279) for robustifying neural networks.

## Background
Given a classification problem of $K$ different classes with a given dataset  $X_N=\{x_n, y_n\}_{n=1}^{N}$ of $N$ datapoints and where $x\in\mathbb{R}^m$, $m$ is the number of dimensions and $y\in\{1,...,K\}$, the task of training a robust neural network $h$ is to solve the robust loss problem or min-max problem with the cross-entropy loss function $\mathcal{L}$ and a perturbation $\delta$ within $\mathcal{U}_p=\\{\delta :||\delta||_p\leq\varepsilon\\}$, thus

$$\min_{\theta}\\frac{1}{N}\sum_{n=1}^{N}\max_{\delta_n\in\mathcal{U}_p}\mathcal{L}(h(x_n+\delta_n,\theta),y_n).$$

The solution to the min-max problem can be approximated with the approximate Robust Upper Bound (aRUB) loss function of the form

$$ \log(\sum_{j=1}^{K}e^{c_{ky}^Th(x,\theta)+\varepsilon||\nabla_xc_{ky}^Th(x,\theta)||_q}),$$

where $e_k$ and $e_y$ are zero vector except for the subscript index location, where the vector takes a value of 1, and $q$ is the dual norm for $p$, i.e. $1/p+1/q=1$. For further details see [Bertsimas et al. (2021)](https://arxiv.org/abs/2112.09279) and [Lundqvist (2023)](https://aaltodoc.aalto.fi/handle/123456789/120190).

## How to use

### Basics
Load the aRUB class to your code:
```
from aRUB import aRUB
```
Define the norm $p$ in $\mathcal{U}_p$ from the available "L1", "L2" and "Linf" as strings, the maximum perturbation (or norm size) $\varepsilon$ as float or double and the amount of classes $K$ as integer. Also specify the device. For example:
```
norm = "L1" 
epsilon = 0.001
n_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Construct and initialize the loss
```
criterion = aRUB(epsilon=epsilon, n_classes=n_classes, device=device, norm=norm)
```
Compute the loss with labels, inputs and the model. The model needs to be given as an argument as it the loss function uses the model's gradients
```
loss, net = criterion(labels, inputs, net)
```
Call backward.() pass normally
```
loss.backward()
```

### Example with Resnet18 and CIFAR10 dataset

Run example.py
```
python3 example.py
```

## Licensing

## References
Bertsimas, D., Boix, X., Carballo, K.V. and Hertog, D.D., 2021. *A robust optimization approach to deep learning*. arXiv preprint arXiv:2112.09279. Available at: [https://arxiv.org/abs/2112.09279](https://arxiv.org/abs/2112.09279)

Lundqvist O., 2023. *A Robust Optimization Approach against Adversarial Attacks on Medical Images*. Aalto University. Available at: [https://aaltodoc.aalto.fi/handle/123456789/120190](https://aaltodoc.aalto.fi/handle/123456789/120190)

# Approximation Rates of a Trained Neural Network

Given a neural network $f_\theta$ and target function $g$, a typical approximation theory result is of the form:

$$\begin{align*}
\inf_{\theta}||f_\theta - g|| \lesssim n(\theta)^{-\alpha}, & \quad g \in K
\end{align*}$$

bounding the error as a function of the number of trainable parameters, $\theta$, at a rate $\alpha$ for some class of functions, $K$. In most such papers, their proofs rely on hand-crafted networks, where the weights and biases are carefully selected. The goal of this project is to find the approximation rate, $\alpha$, for a practical neural network that can be realized via training. Thus far, we've only considered one-dimensional shallow networks:

$$f_\theta(x) := \frac{1}{\sqrt{m}} \sum_{r=1}^m a_r \sigma(x - \theta_r)$$

TODO: add link to arXiv paper...
## Results
For target functions of varying smoothness we use, a Gaussian, a cusp and a step:

$$\begin{align*}
g(x) = 0.5 e^{-12.5 x^2} & \quad g(x) = 1 - \sqrt{|x|} & \quad g(x) = \begin{cases} 
    0 & x\leq 0 \\
    1 & x > 0 
  \end{cases}
\end{align*}$$

TODO: add figures...

### Rates

### Breakpoints

## Installation / Usage

See: [here](./scripts/README.md)

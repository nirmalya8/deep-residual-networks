# Deep Residual Networks: Notes and Implementation 

The main aim of this paper is to ease the training of neural networks, which are substantially deeper, even going up to 1000 layers. 

Say there is a network, which performs quite well on the chosen performance measure. If we add a few more layers at the end of that network, which represent the Identity function($f(x)=x$). So, the performance of this new network should not differ much from the original network as the layers added give out the same output as the input fed to them. Figure 1 shows this representation.  

![Figure 1](./images/fig1.png "Fig 1. Visual Representation of the layers")

<p align="center">Fig 1. Visual representation of the layers</p>

But, do we just train the network and hope the last few layers learn the identity function? Well, no. The authors of the paper go on to introduce a *deep residual learning* framework. 

Let us first answer an important question.

## Does stacking more layers imply better results?

Neural networks, especially the deeper ones have often fallen prey to the vanishing/exploding gradients problem, in which the gradients in the deeper layers either become too close to zero(little to no updates in weights) or become too large(large updates in weights), thus hampering performance.

There is a solution to this problem and the trick lies in normalizing the inputs to the layers. This is mostly taken care by the performing Batch Normalization after taking the activations of one layer, before passing them onto the next.

## The Deep Residual Framework

As mentioned earlier, the deep residual learning framework allows the layers to learn the identity function. Say, the layers need to learn the function $H(x)$, which is evidently defined by $H(x) = x$ ideally. The layers learn a certain function, which is described by $F(x)$. The $x$ which is passed as input, is then added to $F(x)$, thus making it $F(x)+x$ before passing it onto the activation function, as shown in Figure 2. 
<p align="center">
<img width="706" height="624"  src="images/fig2.png")

Fig 2. The blocks which implement residual learning</p>


Thus, $H(x)$ comes out to be: $H(x) = F(x) + x$. $F(x)$, which is generally referred to as the residue is mathematically represented as $F(x) = H(x) - x$. 

Shortuct connections or Skip Connections are used to achieve this. In this case, these shortcut connections are used to perform the identity mapping. As evident from the figure, no extra parameter is added and thus the computational complexity isn't affected either. This network can be trained using Stochastic Gradient Descent with backpropagation. 

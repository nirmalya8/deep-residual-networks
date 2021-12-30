# Deep Residual Networks: Notes and Implementation 

The main aim of this paper is to ease the training of neural networks, which are substantially deeper, even going up to 1000 layers. 

Say there is a network, which performs quite well on the chosen performance measure. If we add a few more layers at the end of that network, which represent the Identity function($f(x)=x$). So, the performance of this new network should not differ much from the original network as the layers added give out the same output as the input fed to them. Fig 1 shows this representation.  

But, do we just train the network and hope the last few layers learn the identity function? Well, no. The authors of the paper go on to introduce a *deep residual learning* framework. 

Let us first answer an important question.

## Does stacking more layers imply better results?
---
Neural networks, especially the deeper ones have often fallen prey to the vanishing/exploding gradients problem, in which the gradients in the deeper layers either become too close to zero(little to no updates in weights) or become too large(large updates in weights), thus hampering performance.

There is a solution to this problem and the trick lies in normalizing the inputs to the layers. This is mostly taken care by the performing Batch Normalization after taking the activations of one layer, before passing them onto the next.

## The Deep Residual Framework
---

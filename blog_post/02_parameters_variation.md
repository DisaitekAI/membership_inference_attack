# Parameter variations

There is a lot of different parameters that affect the result of the
membership inference attack. In this blog post we report the results
of multiple experiences that we ran hoping to shed some light on the
scenarios where such attack can be succesful. Due to the huge number
of parameter combination, each parameter will be explored
independently.

## Number of shadow model

The number of shadow model is clearly an important parameter of this
problem. Prior to running this experiment, it is reasonable to expect
that we will experience disminishing returns as we increase this
value. It is also likely that there is no single "magic number" of
shadow model that will work well for all scenarios.

**Graphs and conclusion**

## Regularization

As explained in the previous blog post, the attack algorithm is based
on the fact that the confidence models have in predictions about
samples coming from the training set is usually higher than on new
ones. It intuitively seems that this behavior could be caused by an
overfitting phenomenon. The natural question that comes next is how
much a reduction in the overfitting of the model impacts the success
of the attack. We explore in this section the effect of various
regularization methods on the membership inference attack.

### Dropout

**Graphs and conclusion**

### Batch normalization

**Graphs and conclusion**

## Architecture differences

### Same network shape but different filter number

What happens is the shadow is more / less prone to overfitting. Maybe
making the shadow model slightly more robust than the target could
help the MIA model focus on important distribution differences (74%
BS).

**Graphs and conclusion**

### Different dropout rates

For the same reason as above, a very slightly more robust shadow may
be beneficial for the overall attack.

**Graphs and conclusion**

### Different architecture of the same type

Seems like a reasonable attack scenario. Let's say we want to attack a
model that performs image recognition. We may not know the specific
architecture of the target model but it is likely that it is in the
CNN family.

**Graphs and conclusion**

### Different type of architecture

We can also ask what happens when the target and the shadow model are
radically different e.g. CNN vs FCN and vice-versa

## Shadow model training methodology

Another interesting parameter to explore is the training methodology
we use for the shadow models. We can either train them as regular
classifiers using a [negative log
likelihood](https://en.wikipedia.org/wiki/Likelihood_function) loss or
using [model distillation](https://arxiv.org/abs/1503.02531). The idea
of model distillation is to train a second model to reproduce the full
probability distribution that a first model outputs for each sample
instead of just trying to have the maximal value at the index
(i.e. predicting the same class).

**Figure to illustrate the concept of model distillation**

(temperature variation maybe)

**Table of results or graph if experimenting with distillation temperatures**


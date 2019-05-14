# Parameter variations

There is a lot of different parameters that affect the result of the
membership inference attack. In this blog post we report the results
of multiple experiences that we ran hoping to shed some light on the
scenarios where such attack can be succesfull.

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

Graphs and conclusion

### Batch normalization

Graphs and conclusion

## Architecture differences

### Same network shape but different filter number

What happens is the shadow is more / less prone to overfitting. Maybe
making the shadow model slightly more robust than the target could
help the MIA model focus on important distribution differences (74%
BS).

### Different dropout rates

For the same reason as above, a very slightly more robust shadow may
be beneficial for the overall attack.

### Different architecture of the same type

Seems like a reasonable attack scenario. Let's say we want to attack a
model that performs image recognition. We may not know the specific
architecture of the target model but it is likely that it is in the
CNN family.

### Different type of architecture



# Membership inference attack

A lot of energy is currently being deployed in ensuring machine
learning models do not leak confidential informations to the public.
In this series of blog post, we study a "membership inferenced attack"
i.e. trying to tell whether a specific sample was part of the training
set of the model or not. In particular, we will explain and study the
method described in the paper [Membership Inference Attacks Against
Machine Learning Models](https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf).

The first blog post of this series is an introduction to the
attack methodology, prerequisites, consequences and countermeasures

## Consequences

- Data leaking
- GDPR non-conformity

## Attack methodology

### Prerequisites

- Having access to the confidence the model have in its prediction.
- We are able to run the model on as many samples as we want.

### Attack concept

- We want to exploit the fact that the machine learning models are
  usually more confident while predicting on samples from the training
  than on new samples.
- We build a collection of models (shadow models) that mimic the
  behavior of the target. As we are training these models ourselves,
  we know for each whether a specific sample is part of their training
  set or not. With this information, we can observe how the confidence
  of the shadow model is affected by the fact that the sample belongs
  to the training set or not. Finally, we can train a new ML model
  (MIA model) that predicts based on the confidence of the target
  model whether a sample is part of the training set or not.

### Attack algorithm

- Build a synthetic training dataset by running many inference on the
  target dataset and using its predictions as labels.
- Pick a shadow model architecture
- Train a collection of shadow models using the synthetic dataset
- Build a dataset for the MIA model using this set of models
- Train the MIA model
- Profit.

## Countermeasures

- Hiding model confidence.
- Differential privacy techniques.

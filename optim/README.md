# Select the models to be on memory

This is an optimization problem that selects which models to include in an ensemble.
It decides to select a subset of models from a larger set, given certain constraints and objectives.


## Model Selection

### Input:

- $M = \{1, 2, \dots, N \}$: Set of models
- $K = \{1, 2, \dots, L \}$: Set of classes

#### Parameters

- $S_m \in \mathbb{N}^+$: Size of model $m$, for $m \in M$
- $P$: list of allocation priority, where
    - $P[k] = p_k$ for $k \in K$
    - $p_k \in (0, 1]$: priority of class $k \in K$
    - $\sum_{k \in K} p_k \leq 1$
- $W$: List of weights for each model $m$ for each class $k$:
    - $W[m, k] = w_{m,k}$ for $m \in M, k \in K$
    - $w_{m,k} \in \mathbb{R}^{+}$: Contribution weight of model $m$ for class $k$, for $m \in M, k \in L$.
- $M_e$: Total memory available
- $R \subseteq L$: Ordered set of prioritized classes


### Decision Variables:

- $x_m \in {0, 1}$: Binary variable indicating whether model $m$ is selected ($x_m = 1$) or not ($x_m = 0$), for $m \in M$

- $y_{m,k} \in {0, 1}$: Binary variable indicating whether model $m$ is selected for class $k$ ($y_{m,k} = 1$) or not ($y_{m,k} = 0$), for $m \in M, k \in K$




### Objective Function:

Maximize the weighted contribution of selected models, prioritizing classes in R:


$$
\text{Maximize} \quad Z = \sum_{k \in K} p_k \sum_{m \in M} \frac{1}{\sum_{j \in M} w_{j, c}} \cdot w_{m, k} \cdot y_{m, k}
$$


Thus, the selected models are:

$$
\Theta = \{m | x_m = 1 \}
$$


### Constraints:

#### Memory Constraint:

The total size of selected models cannot exceed the available memory:

$$
\sum_{m \in M} S_m \cdot x_m \leq M_e
$$

#### Model Selection for Class:

Model $m$ can be selected for class $k$ only if it is selected in the ensemble:

$$
y_{m,k} \leq x_m \quad \forall m \in M, k \in K
$$

If one model m is selected, it means at least one class c of model m is also selected:

$$
\sum_{k \in K} y_{m,k} \ge x_m \quad \forall m \in M
$$


## The Accuracy Contribution

- We have N models, each classifying L classes.

Thus, $w_{m,c}$ is a "contribution weight".
Let's say model m outputs a probability (or score) $p_{m,c}$ for a given sample belonging to class c. The ensemble's output for class c, denoted as $P_c$, is calculated by taking a weighted sum of the probabilities from the models selected for class c:

$$P_c = \sum_{m \in \Theta} w_{m, c} \cdot p_{m,c}$$


The final classification is typically determined by selecting the class with the highest $P_c$ value.

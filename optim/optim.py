"""

This script implements a Mixed Integer Linear Programming (MILP) model selection algorithm
for a machine learning ensemble. The goal is to select a subset of models that maximizes the
weighted performance across multiple classes while adhering to a memory constraint.


Example:
python optim/optim.py


**Note**:
- The script uses the `mip` library for MILP optimization. Tested with mip==1.15.0.
"""

from mip import Model, xsum, maximize
from mip import BINARY, OptimizationStatus


def solve_model_selection_milp(S, P, W, Me, max_seconds=600, verbose=False):
    """
    Solves the model selection problem using Mixed Integer Linear Programming (MILP).

    Args:
        S (list): List of model sizes, where S[m] is the size of model m.
        P (list): List of priority class, where A[k] = p_k
            - p_k (float): Priority of class c_k. Sum of all p_k should be 1.
        W (nested list): List of tuples for each class c_k, where W_c_k = (v_{j, c_k}, m_{j, c_k}).
            - W_{c_k} (list): List of (weight, model) tuples for class c_k, sorted in
              descending order of weight.  Each tuple is (v_{j, c_k}, m_{j, c_k}).
        Me (int): Total memory available.

    Returns:
        tuple: A tuple containing:
            - selected_models (set): Set of model indices to include in the ensemble.
            - class_model_mapping (dict): Dictionary where each key is a class and the
              value is a list of model indices selected for that class.
              Returns None if the problem is infeasible.
    """
    M = len(S)  # Number of models
    K = len(P) # Number of classes

    assert len(W) == len(P), f"Number of classes must match number of priorities: {K}"
    assert all([len(w) == len(S) for w in W]), f"Number of models must match number of weights: #{M}"

    # Normalize priorities
    P = [x / sum(P) for x in P]

    # 1. Create the MILP model
    model = Model("model_selection")
    model.verbose = 1 if verbose else 0

    # 2. Decision Variables
    x = [model.add_var(var_type=BINARY, name=f"x_{m}") for m in range(M)]
    y = [[model.add_var(var_type=BINARY, name=f"y_{m}_{c}") for c in range(K)] for m in range(M)]

    # 3. Objective Function
    objective_terms = []
    for k in range(K):
        p_k = P[k]
        W_k = sum(W[k])  # to normalize the weights
        for m, w_jk in enumerate(W[k]):
            objective_terms.append(p_k * w_jk / W_k * y[m][k])
    model.objective = maximize(xsum(objective_terms))

    # 4. Constraints
    # Memory Constraint
    model.add_constr(xsum(S[m] * x[m] for m in range(M)) <= Me, "memory_constraint")

    # Model Selection for Class
    for m in range(M):
        # If the model is selected, it must be selected for at least one class
        for k in range(K):
            model.add_constr(y[m][k] <= x[m], f"class_selection_{m}_{k}")

        # Ensure at least one model is selected for each class if the model is used
        model.add_constr(sum([y[m][k] for k in range(K)]) >= x[m], f"class_selection_{m}")


    # Solve the model
    status = model.optimize(max_seconds=max_seconds)

    if status == OptimizationStatus.OPTIMAL:  # OptimizationStatus.OPTIMAL
        selected_models = {m for m in range(M) if x[m].x >= 0.9}  # Use a tolerance
        prioritized_classes = {c for c in range(K) for m in range(M) if y[m][c] >= 0.9}

        if verbose:
            print(f"Status: {model.status}")
            print(f"Number of solutions: {model.num_solutions}")
            print(f"Selected models: {selected_models}")
            print(f"Prioritized classes: {prioritized_classes}")

            # print("Selected models:", [int(x[m].x) for m in range(M)])
            print("Class allocation:")
            for c in range(K):
                print(f"{c:2d}: ", " ".join([str(int(y[m][c].x)) for m in range(M)]))

        return selected_models, prioritized_classes
    else:
        print(f"Problem is {model.status}")
        return None, None


if __name__ == "__main__":
    # Example Usage
    P = [10, 5, 1]

    W = [
        [0.1, 0.1, 0.4, 0.1, 0],    # Class 0
        [0.1, 0.1, 0.3, 0.3, 0.4],  # Class 1
        [0.1, 0.1, 0, 0.2, 0.1],    # Class 2
    ]

    # Model sizes
    S = [20, 30, 40, 50, 35]  # Model sizes

    # Memory constraint
    # Me = sum(S)  # Total memory available, guarantees that the problem is feasible
    Me = 55  # Restricted memory

    selected_models, prioritized_classes = solve_model_selection_milp(S, P, W, Me)

    if selected_models is not None:
        print(f"Number of models: {len(S)}")
        print(f"Number of classes: {len(P)}")

        print("Contribution weights:")
        for i, row in enumerate(W):
            print(f"{i}: {', '.join(f'{x:.2f}' for x in row)}")

        print(f"Assigned priority: {P}")
        print(f"Model sizes: {S} <= {Me}")
        # solution
        print("\nSelected models:", selected_models)
        print("Class-Model Mapping:", prioritized_classes)
    else:
        print("No feasible solution found.")

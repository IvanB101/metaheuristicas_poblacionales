import numpy as np
import matplotlib.pyplot as plt


def objective(x) -> float:
    return np.add.reduce(x**2) / len(x)


def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


def check_bounds(mutated, bounds):
    mutated_bound = [
        np.clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))
    ]
    return mutated_bound


def crossover(mutated, target, dims, cr):
    p = np.random.random(dims)
    # generate trial vector by binomial crossover
    trial = np.array([mutated[i] if p[i] < cr else target[i] for i in range(dims)])
    return trial


def differential_evolution(
    D: int,
    Np: int = 500,
    F: float = 0.5,
    Cr: float = 0.7,
    limits: tuple[float, float] = (-5.0, 5.0),
    iter: int = 100,
    verbose: bool = True,
):
    """
    Parameters
    - pop_size (int): size of the population
    - bounds (tuple[float, float]): upper and lower limits
    - iter (int): number of iterations
    - F (float): scale factor for mutation
    - Cr (float): crossover rate for recombination
    - verbose (bool): if true reports each time the best fitness metric is improved
    """
    bounds = np.array([(limits[0], limits[1]) for _ in range(D)])
    # initialise population of candidate solutions randomly within the specified bounds
    population = bounds[:, 0] + (
        np.random.random((Np, len(bounds))) * (bounds[:, 1] - bounds[:, 0])
    )
    obj_all = [objective(ind) for ind in population]
    # find the best performing vector of initial population
    best_vector = population[np.argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    generations = []

    for i in range(iter):
        for j in range(Np):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(Np) if candidate != j]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]

            mutated = mutation([a, b, c], F)
            mutated = check_bounds(mutated, bounds)

            trial = crossover(mutated, population[j], len(bounds), Cr)

            if (obj_trial := objective(trial)) < obj_all[j]:
                population[j] = trial
                obj_all[j] = obj_trial

        best_obj = min(obj_all)
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = population[np.argmin(obj_all)]
            prev_obj = best_obj
            generations.append(best_obj)
            # report progress at each iteration
            if verbose:
                print(
                    "Iteration: %d f([%s]) = %.5f"
                    % (i, np.around(best_vector, decimals=5), best_obj)
                )

    return best_vector, best_obj, generations


dimensions = [10, 50, 100]
colors = ["#2187bb", "#149c1b", "#3916a1"]
for dimension, color in zip(dimensions, colors):
    best_vector, best_obj, generations = differential_evolution(
        dimension, verbose=False
    )
    print("\nSolution: f([%s]) = %.5f" % (np.around(best_vector, decimals=5), best_obj))

    plt.plot(generations, color=color, label=("n: {}".format(dimension)))

plt.xlabel("Improvement Number")
plt.ylabel("Evaluation f(x)")
plt.legend()
plt.show()

from typing import Any
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
    report_inter: int = 20,
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
    generations: list[list[Any]] = [[], [], []]

    for i in range(iter):
        if i % report_inter == 0:
            mean = np.mean(obj_all)
            max = np.max(obj_all)
            min = np.min(obj_all)
            generations[0].append(min)
            generations[1].append(mean)
            generations[2].append(max)
            # report progress at each iteration
            if verbose:
                print(
                    "Iteration: %{} min: {}, mean: {}, max: {}".format(
                        i, min, mean, max
                    )
                )

        for j in range(Np):
            # reporting

            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(Np) if candidate != j]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]

            mutated = mutation([a, b, c], F)
            mutated = check_bounds(mutated, bounds)

            trial = crossover(mutated, population[j], len(bounds), Cr)

            if (obj_trial := objective(trial)) < obj_all[j]:
                population[j] = trial
                obj_all[j] = obj_trial

    best_vector = population[np.argmin(obj_all)]
    best_obj = np.min(obj_all)

    return best_vector, best_obj, generations


dimensions = [10, 50, 100]
colors = ["#1187bb", "#148c1b", "#3814a1"]
for dimension, color in zip(dimensions, colors):
    best_vector, best_obj, [mins, means, maxs] = differential_evolution(
        dimension,
        verbose=False,
    )
    print("\nSolution: f([%s]) = %.5f" % (np.around(best_vector, decimals=5), best_obj))

    plt.plot(mins, "--", color=color)
    plt.plot(means, color=color, label=("n: {}".format(dimension)))
    plt.plot(maxs, "--", color=color)

plt.xlabel("Sample point")
plt.ylabel("Evaluation f(x)")
plt.legend()
plt.show()

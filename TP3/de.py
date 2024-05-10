from random import random, choice
from numpy import argmin, around, clip, asarray
import matplotlib as pyplot


# define objective function
def obj(x):
    return sum(x**2)


def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [
        clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))
    ]
    return mutated_bound


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = random(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def differential_evolution(
    pop_size: int,
    bounds=asarray([(-5.0, 5.0), (-5.0, 5.0)]),
    iter: int = 100,
    F: float = 0.5,
    cr: float = 0.7,
):
    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (random(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    # evaluate initial population of candidate solutions
    obj_all = [obj(ind) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    evolution = []
    # run iterations of the algorithm
    for i in range(iter):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = obj(pop[j])
            # compute objective function value for trial vector
            obj_trial = obj(trial)
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            evolution.append(best_vector)
            # report progress at each iteration
            print(
                "Iteration: %d f([%s]) = %.5f"
                % (i, around(best_vector, decimals=5), best_obj)
            )

    return evolution, best_vector, best_obj


evolution, best_vector, best_obj = differential_evolution(10)

# perform differential evolution
solution = differential_evolution(pop_size, bounds, iter, F, cr)
print("\nSolution: f([%s]) = %.5f" % (around(solution[0], decimals=5), solution[1]))

# line plot of best objective function values
pyplot.plot(evolution, ".-")
pyplot.xlabel("Improvement Number")
pyplot.ylabel("Evaluation f(x)")
pyplot.show()

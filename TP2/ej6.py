from random import seed, randint
import numpy as np
import matplotlib.pyplot as plt


def create_instance(chromosome_length):
    weights = np.random.uniform(10, 100, chromosome_length)
    values = np.random.uniform(20, 50, chromosome_length)
    capacity = np.add.reduce(weights) / 2

    return weights, values, capacity


def create_starting_population(individuals, chromosome_length):
    population = np.zeros((individuals, chromosome_length))
    for i in range(individuals):
        ones = randint(0, chromosome_length)
        population[i, 0:ones] = 1
        np.random.shuffle(population[i])

    return population


def calculate_fitness(population, weights, values, capacity):
    return np.array(
        [
            -1 if sum(individual * weights) > capacity else sum(individual * values)
            for individual in population
        ]
    )


def select_individual_by_tournament(population, scores):
    population_size = len(scores)

    fighter_1 = randint(0, population_size - 1)
    fighter_2 = randint(0, population_size - 1)

    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]

    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2

    return population[winner, :]


def breed_by_crossover(parent_1, parent_2):
    chromosome_length = len(parent_1)

    crossover_point = randint(1, chromosome_length - 1)

    child_1 = np.hstack((parent_1[0:crossover_point], parent_2[crossover_point:]))

    child_2 = np.hstack((parent_2[0:crossover_point], parent_1[crossover_point:]))

    return child_1, child_2


def randomly_mutate_population(population, mutation_probability):
    random_mutation_array = np.random.random(size=(population.shape))

    random_mutation_boolean = random_mutation_array <= mutation_probability

    population[random_mutation_boolean] = np.logical_not(
        population[random_mutation_boolean]
    )

    return population


def run(
    chromosome_length,
    population_size,
    maximum_generation,
    instance,
    initial_population,
    elite,
    s,
):
    seed(s)
    best_score_progress = []
    weights, values, capacity = instance
    population = initial_population

    scores = calculate_fitness(population, weights, values, capacity)
    best_score = np.max(scores) / chromosome_length * 100
    print("Starting best score, % target: ", best_score)

    best_score_progress.append(best_score)

    for _ in range(maximum_generation):
        new_population = []

        for _ in range(int(population_size * (1 - elite) / 2)):
            parent_1 = select_individual_by_tournament(population, scores)
            parent_2 = select_individual_by_tournament(population, scores)
            child_1, child_2 = breed_by_crossover(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)

        population = list(zip(population, scores))
        population.sort(key=lambda x: -x[1])
        population = [i for i, _ in population[: int(population_size * elite)]]

        mutation_rate = 0.002
        new_population = randomly_mutate_population(
            np.array(new_population), mutation_rate
        )

        if elite > 0:
            population = np.concatenate((np.array(population), new_population))
        else:
            population = new_population

        scores = calculate_fitness(population, weights, values, capacity)
        best_score = np.max(scores) / chromosome_length * 100
        best_score_progress.append(best_score)

    return best_score, best_score_progress


# genetic algorith parameters
chromosome_length = 150
population_size = 500
maximum_generation = 500
# a common seed and starting population for better consistency
s = 0
initial_population = create_starting_population(population_size, chromosome_length)
instance = create_instance(chromosome_length)

# elite fractions that pass to the next generation
elite_portions = [0, 0.01, 0.02, 0.1, 0.2]
colors = ["#2187bb", "#149c1b", "#3916a1", "#cf6a3c", "#8c0e21"]
for portion, color in zip(elite_portions, colors):
    best_score, best_score_progress = run(
        chromosome_length,
        population_size,
        maximum_generation,
        instance,
        initial_population,
        portion,
        s,
    )

    print("End best score with {}% elite: {}".format(portion * 100, best_score))
    plt.plot(
        best_score_progress, color=color, label=("{}% elite".format(portion * 100))
    )


plt.xlabel("Generation")
plt.ylabel("Best score")
plt.legend()
plt.show()

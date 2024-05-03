from random import seed, randint, shuffle, uniform
import numpy as np
import matplotlib.pyplot as plt


def create_instance(chromosome_length):
    distances = np.zeros((chromosome_length, chromosome_length))

    for i in range(0, chromosome_length):
        for j in range(i + 1, chromosome_length):
            distances[i][j] = distances[j][i] = uniform(1, 100)

    return distances


def create_starting_population(individuals, chromosome_length):
    base = list(range(chromosome_length))
    population = []

    for _ in range(individuals):
        shuffle(base)
        population.append(base[:])

    return np.array(population)


def calculate_fitness(population, distances):
    chromosome_length = len(population[0])

    return np.array(
        [
            (
                float("inf")
                if len(individual) != len(set(individual))
                else sum(
                    [
                        distances[individual[i - 1]][individual[i]]
                        for i in range(1, chromosome_length)
                    ]
                )
                + distances[individual[chromosome_length - 1]][individual[0]]
            )
            for individual in population
        ]
    )


def select_individual_by_tournament(population, scores):
    population_size = len(scores)

    fighter_1 = randint(0, population_size - 1)
    fighter_2 = randint(0, population_size - 1)

    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]

    if fighter_1_fitness < fighter_2_fitness:
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


def breed_by_crossover_2(parent_1, parent_2):
    chromosome_length = len(parent_1)

    crossover_point = randint(1, chromosome_length - 1)

    temp_1 = parent_1[crossover_point:]
    last_1 = []
    temp_2 = parent_2[crossover_point:]
    last_2 = []

    for i in parent_2:
        if i in temp_1:
            last_2.append(i)

    for i in parent_1:
        if i in temp_2:
            last_1.append(i)

    child_1 = np.hstack((parent_1[:crossover_point], last_2))
    child_2 = np.hstack((parent_2[:crossover_point], last_1))

    return child_1, child_2


def randomly_mutate_population(population, mutation_probability):
    chromosome_length = len(population[0])

    for individual in population:
        for _ in range(int(mutation_probability * chromosome_length)):
            index_1 = randint(0, chromosome_length - 1)
            index_2 = randint(0, chromosome_length - 1)
            temp = individual[index_1]
            individual[index_1] = individual[index_2]
            individual[index_2] = temp

    return population


def run(
    chromosome_length,
    population_size,
    maximum_generation,
    instance,
    initial_population,
    mutation_rate,
    elite,
    s,
    v=1,
):
    seed(s)
    best_score_progress = []
    distances = instance
    population = initial_population

    scores = calculate_fitness(population, distances)
    best_score = np.min(scores) / chromosome_length * 100
    print("Starting best score, % target: ", best_score)

    best_score_progress.append(best_score)
    for _ in range(maximum_generation):
        new_population = []

        for _ in range(int(population_size * (1 - elite) / 2)):
            parent_1 = select_individual_by_tournament(population, scores)
            parent_2 = select_individual_by_tournament(population, scores)
            if v == 1:
                child_1, child_2 = breed_by_crossover(parent_1, parent_2)
            else:
                child_1, child_2 = breed_by_crossover_2(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)

        population = list(zip(population, scores))
        population.sort(key=lambda x: x[1])
        population = [i for i, _ in population[: int(population_size * elite)]]

        new_population = randomly_mutate_population(
            np.array(new_population), mutation_rate
        )

        if elite > 0:
            population = np.concatenate((np.array(population), new_population))
        else:
            population = new_population

        scores = calculate_fitness(population, distances)
        best_score = np.min(scores) / chromosome_length * 100
        best_score_progress.append(best_score)

    return best_score, best_score_progress


# genetic algorith parameters
chromosome_length = 75
population_size = 500
maximum_generation = 5000
mutation_rate = 0.02
# a common seed and starting population for better consistency
s = 0
initial_population = create_starting_population(population_size, chromosome_length)
instance = create_instance(chromosome_length)

elite_portions = [0, 0.01, 0.05, 0.1, 0.2]
colors = ["#2187bb", "#149c1b", "#3916a1", "#cf6a3c", "#8c0e21"]

# elite fractions that pass to the next generation
for portion, color in zip(elite_portions, colors):
    best_score, best_score_progress = run(
        chromosome_length,
        population_size,
        maximum_generation,
        instance,
        initial_population,
        mutation_rate,
        portion,
        s,
    )

    print("End best score with {}% elite: {}".format(portion * 100, best_score))
    plt.plot(
        best_score_progress, color=color, label=("{}% elite".format(portion * 100))
    )

colors = ["#9187bb", "#147c1b", "#301601", "#0f6a3c", "#600921"]
for portion, color in zip(elite_portions, colors):
    best_score, best_score_progress = run(
        chromosome_length,
        population_size,
        maximum_generation,
        instance,
        initial_population,
        mutation_rate,
        portion,
        s,
        2,
    )

    print("End best score v2 with {}% elite: {}".format(portion * 100, best_score))
    plt.plot(
        best_score_progress, color=color, label=("f2 {}% elite".format(portion * 100))
    )


plt.xlabel("Generation")
plt.ylabel("Best score")
plt.legend()
plt.show()

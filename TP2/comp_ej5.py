import random

POPULATION_SIZE = 100

GENES = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}"""

TARGET = "Probando el primer algoritmo genetico"


class Original(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(cls):
        global TARGET
        gnome_len = len(TARGET)
        return [cls.mutated_genes() for _ in range(gnome_len)]

    def mate(self, par2):

        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):

            prob = random.random()

            if prob < 0.45:
                child_chromosome.append(gp1)

            elif prob < 0.90:
                child_chromosome.append(gp2)

            else:
                child_chromosome.append(self.mutated_genes())

        return Original(child_chromosome)

    def cal_fitness(self):
        global TARGET
        fitness = 0
        for gs, gt in zip(self.chromosome, TARGET):
            if gs != gt:
                fitness += 1
        return fitness


class Modified(object):

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(cls):
        global TARGET
        gnome_len = len(TARGET)
        return [cls.mutated_genes() for _ in range(gnome_len)]

    def mate(self, par2):
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            prob = random.random()

            if prob < 0.45:
                child_chromosome.append(gp1)

            elif prob < 0.90:
                child_chromosome.append(gp2)

            else:
                child_chromosome.append(self.mutated_genes())

        return Modified(child_chromosome)

    def cal_fitness(self):
        global TARGET
        fitness = 0
        for gs, gt in zip(self.chromosome, TARGET):
            fitness += abs(ord(gs) - ord(gt))
        return fitness


def run(imp):
    global POPULATION_SIZE

    generation = 1

    found = False
    population = []

    for _ in range(POPULATION_SIZE):
        gnome = imp.create_gnome()
        population.append(imp(gnome))

    while not found:

        population = sorted(population, key=lambda x: x.fitness)

        if population[0].fitness <= 0:
            found = True
            break
        new_generation = []

        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])

        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation
        generation += 1

    return generation


test_number = 30

orig_gens = [run(Original) for _ in range(test_number)]
mod_gens = [run(Modified) for _ in range(test_number)]

print("Generation numbers: {}".format(orig_gens))
print("Generation numbers: {}".format(mod_gens))

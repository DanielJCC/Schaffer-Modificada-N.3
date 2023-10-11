import math
import random
import matplotlib.pyplot as plt
import statistics

#T = 4
alpha = 0.998
stopping_temperature = 1e-8
stopping_iter = 20000
best_solution = None


# Fitness o función objetivo
def schaffer(x, y):
    return 0.5 + ((math.sin(math.cos(abs(x**2-y**2)))**2 - 0.5) / (1 + 0.001*(x**2 + y**2))**2)

# Enfriamiento simulado
def simulated_annealing(initial_temperature, cooling_rate, num_iterations):
    fitness_list = []
    current_solution = [random.uniform(-100, 100), random.uniform(-100, 100)]
    current_energy = schaffer(current_solution[0], current_solution[1])

    temperature = initial_temperature
    iteration = 0

    while temperature >= stopping_temperature and iteration < stopping_iter:
        new_solution = [current_solution[0] + random.uniform(-1, 1), current_solution[1] + random.uniform(-1, 1)]
        new_energy = schaffer(new_solution[0], new_solution[1])

        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / temperature):
            current_solution = new_solution
            current_energy = new_energy
        fitness_list.append(current_energy)
        temperature *= alpha
        iteration+=1

    # for i in range(num_iterations):
    #     temperature = initial_temperature * math.exp(-cooling_rate * i)
    #     new_solution = [current_solution[0] + random.uniform(-1, 1), current_solution[1] + random.uniform(-1, 1)]
    #     new_energy = schaffer(new_solution[0], new_solution[1])
        
    #     if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / temperature):
    #         current_solution = new_solution
    #         current_energy = new_energy

    #     fitness_list.append(current_energy)
    print("Best fitness obtained: ",current_energy)
    improvement = 100 * (fitness_list[0] - current_energy) / (fitness_list[0])
    print(f"Improvement over greedy heuristic: {improvement: .2f}%")
    print("Optimal solution: ",current_solution)
    return current_solution, current_energy, fitness_list

def Graficar(fitness_list):
    plt.plot([i for i in range(len(fitness_list))], fitness_list)
    plt.ylabel("Fitness")
    plt.xlabel("Iteration")
    plt.show()

#simulated_annealing(6.70)
initial_temperature = 10000
cooling_rate = 0.003
num_iterations = 20000
best_solutions = []

# Ejecutar el algoritmo de simulated annealing
for i in range(20):
    print("=================== Iteración",i,"===================")
    optimal_solution, optimal_energy, fitness_list = simulated_annealing(initial_temperature, cooling_rate, num_iterations)
    best_solutions.append(optimal_energy)
    if i == 0:
        Graficar(fitness_list)

print("Costo total promedio =", statistics.mean(best_solutions))
varianza = statistics.variance(best_solutions)
print("Varianza: ", varianza)
print("Mejor solución encontrada: %f" % (min(best_solutions)))
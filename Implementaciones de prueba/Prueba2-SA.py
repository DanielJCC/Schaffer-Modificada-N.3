import math
import random
import matplotlib.pyplot as plt
import statistics
import timeit

# Fitness o función objetivo
def schaffer(x, y):
    return 0.5 + ((math.sin(math.cos(abs(x**2-y**2)))**2 - 0.5) / (1 + 0.001*(x**2 + y**2))**2)

# Enfriamiento simulado
def simulated_annealing(initial_temperature, cooling_rate, num_iterations):
    fitness_list = []
    current_solution = [random.uniform(-100, 100), random.uniform(-100, 100)]
    current_energy = schaffer(current_solution[0], current_solution[1])
    sol_iter = 0

    for i in range(num_iterations):
        temperature = initial_temperature * math.exp(-cooling_rate * i)
        new_solution = [current_solution[0] + random.uniform(-1, 1), current_solution[1] + random.uniform(-1, 1)]
        new_energy = schaffer(new_solution[0], new_solution[1])
        
        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / temperature):
            current_solution = new_solution
            current_energy = new_energy
            sol_iter = i

        fitness_list.append(current_energy)
    print("Best fitness obtained: ",current_energy)
    improvement = 100 * (fitness_list[0] - current_energy) / (fitness_list[0])
    print(f"Improvement over greedy heuristic: {improvement: .2f}%")
    print("Optimal solution: (",round(current_solution[0],2),",",round(current_solution[1],2),")")
    return current_solution, current_energy, fitness_list, sol_iter

# Graficar para observar convergencia
def Graficar(fitness_list):
    plt.plot([i for i in range(len(fitness_list))], fitness_list)
    plt.ylabel("Fitness")
    plt.xlabel("Iteration")
    plt.show()

# Inicializar los parámetros
initial_temperature = 40
cooling_rate = 0.003
num_iterations = 30000
best_solutions = []

# Ejecutar el algoritmo de simulated annealing
for i in range(20):
    print("=================== Iteración",i+1,"===================")
    start_time = timeit.default_timer() 
    optimal_solution, optimal_energy, fitness_list, nIter= simulated_annealing(initial_temperature, cooling_rate, num_iterations)
    stop_time = timeit.default_timer()
    end_time = stop_time - start_time
    print("running_time: ",format(end_time, '.8f'))
    print('Encontró la solución en la iteración: %d' % nIter)
    best_solutions.append(optimal_energy)
    if i == 0:
        Graficar(fitness_list)

print("===============================================")
print("Costo total promedio =", statistics.mean(best_solutions))
varianza = statistics.variance(best_solutions)
print("Varianza: ", varianza)
print("Mejor solución encontrada: %f" % (min(best_solutions)))
## SI SIRVE ESTA Y DA IGU
from simanneal import Annealer
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics
import timeit

# Definir la función Schaffer
def schaffer(x, y):
    return 0.5 + ((math.sin(math.cos(abs(x**2-y**2)))**2 - 0.5) / (1 + 0.001*(x**2 + y**2))**2)

# Clase para el problema de optimización de Schaffer
class SchafferProblem(Annealer):
    def __init__(self, initial_state=None, load_state=None):
        self.fitness_list = []
        super().__init__(initial_state, load_state)
        
    def move(self):
        # Modificar aleatoriamente las variables x y y (Primera forma)
        #   self.state = [self.state[0] + random.uniform(-1, 1), self.state[1] + random.uniform(-1, 1)]
        
        # Modificar aleatoriamente las variables x y y (Segunda forma, la copie del repositorio)
        self.update()
    
    def epson_vector(self, guess, mu = 0, sigma = 1):
        epson = np.zeros(2)
        for j in range(0, 2):
            random_value = np.random.normal(mu, sigma, 1)[0]
            epson[j] = float(random_value)
        return epson
    
    def update(self):
        state = np.array(self.state)
        updated_solution = np.copy(state)
        epson = self.epson_vector(guess=state)
        for j in range(0, 2):
            if (state[j] + epson[j] > 100):
                updated_solution[j] = random.uniform(-100, 100)
            elif (state[j] + epson[j] < -100):
                updated_solution[j] = random.uniform(-100, 100)
            else:
                updated_solution[j] = state[j] + epson[j] 
        self.state = updated_solution

    def energy(self):
        # Calcular el valor de la función Schaffer para el estado actual
        self.fitness_list.append(self.best_energy)
        return schaffer(self.state[0], self.state[1])

best_solutions = []
for i in range(20):
    # Crear una instancia del problema de Schaffer
    best_iteration = 0
    initial_state = [random.uniform(-100, 100), random.uniform(-100, 100)]
    schaffer_problem = SchafferProblem(initial_state)

    # Configurar los parámetros del algoritmo de Simulated Annealing
    schaffer_problem.set_schedule({'tmax': 40, 'tmin': 1e-11, 'steps': 20000, 'updates': 0.00095})

    # Ejecutar el algoritmo de Simulated Annealing
    start_time = timeit.default_timer() 
    best_solution, min_energy = schaffer_problem.anneal()
    stop_time = timeit.default_timer()
    end_time = stop_time - start_time
    best_iteration = schaffer_problem.fitness_list.index(min_energy)
    print("============================= Iteración",i+1,"=============================")
    print("Solución óptima encontrada:")
    print("x:", best_solution[0])
    print("y:", best_solution[1])
    print("Valor de la función Schaffer en la solución óptima:", min_energy)
    print('Encontró la solución en la iteración: %d' % best_iteration)
    print("running_time: ",format(end_time, '.8f'),"seg")
    best_solutions.append(min_energy)

    if i == 0: 
        plt.plot([i for i in range(len(schaffer_problem.fitness_list))], schaffer_problem.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()

print("===============================================")
print("Costo total promedio =", statistics.mean(best_solutions))
varianza = statistics.variance(best_solutions)
print("Varianza: ", varianza)
print("Mejor solución encontrada: %f" % (min(best_solutions)))
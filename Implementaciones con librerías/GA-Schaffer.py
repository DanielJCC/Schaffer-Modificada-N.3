import geneticalgorithm2 as ga
import numpy as np
import timeit
import matplotlib.pyplot as plt
import statistics
from tabulate import tabulate

# Definir la función Schaffer
def schaffer(X):
    return 0.5 + ((np.sin(np.cos(abs(X[0]**2-X[1]**2)))**2 - 0.5) / (1 + 0.001*(X[0]**2 + X[1]**2))**2)

#Definir los límites de la función
var_bound=np.array([[-100,100]]*2)

#Configurar el algoritmo genético
model = ga.geneticalgorithm2(schaffer, dimension = 2, 
                variable_type='real', 
                 variable_boundaries = var_bound,
                 function_timeout = 10,
                 algorithm_parameters=ga.AlgorithmParams(
                     max_num_iteration = None,
                     population_size = 100,
                     mutation_probability = 0.1,
                     mutation_discrete_probability = None,
                     elit_ratio = 0.01,
                     parents_portion = 0.3,
                     crossover_type = 'uniform',
                     mutation_type = 'uniform_by_center',
                     mutation_discrete_type = 'uniform_discrete',
                     selection_type = 'roulette',
                     max_iteration_without_improv = None
                     )
            )

# Ejecutar el algoritmo genético
best_solutions = []
table = []
for i in range(30):
    start_time = timeit.default_timer() 
    result = model.run(disable_printing=True,no_plot=True)
    stop_time = timeit.default_timer()
    end_time = stop_time - start_time
    # print("\nrunning_time: ",format(end_time, '.8f'),"seg")
    # print("Best solution:",result.variable)
    # print("Fitness:",result.score)
    # print("Solución encontrada en la iteración:", model.report.index(result.score))
    improvement = 100 * (model.report[0] - result.score) / (model.report[0])
    # print(f"Improvement over greedy heuristic: {improvement: .2f}%")
    table.append([i+1,result.variable[0],result.variable[1],result.score,model.report.index(result.score),improvement,end_time])
    best_solutions.append(result.score)
    if i == 0:
        plt.plot([i for i in range(len(model.report))], model.report)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()
print("\n")
print(tabulate(table, headers=["Iteration", "x1","x2","Best fitness","Best iteration","Improvement","Running time"],tablefmt="outline"))
print("===============================================")
print("Costo total promedio =", statistics.mean(best_solutions))
varianza = statistics.variance(best_solutions)
print("Varianza: ", varianza)
print("Mejor solución encontrada: %f" % (min(best_solutions)))
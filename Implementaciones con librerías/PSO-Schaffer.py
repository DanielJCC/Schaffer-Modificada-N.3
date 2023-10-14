import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
import timeit
import numpy as np
import statistics
from pyswarms.utils.plotters.formatters import Mesher, Designer
from tabulate import tabulate
import pandas as pd
def schaffer(x):
    x1, x2 = x[:, 0], x[:, 1]
    return 0.5 + ((np.sin(np.cos(abs(x1**2-x2**2)))**2 - 0.5) / (1 + 0.001*(x1**2 + x2**2))**2)

# Definir límites inferiores y superiores para las variables x1 y x2
lower_bound = np.array([-100, -100])  # límites inferiores para x1 y x2
upper_bound = np.array([100, 100])    # límites superiores para x1 y x2
bounds = (lower_bound, upper_bound)
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO

best_solutions = []
best_times = []
best_iterations = []
table = []
for i in range(30):
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
    print("=========================== Iteración",i+1,"===========================")
    start_time = timeit.default_timer()
    best_cost, best_pos = optimizer.optimize(schaffer, iters=100)
    stop_time = timeit.default_timer()
    end_time = stop_time - start_time
    # print("running_time: ",format(end_time, '.8f'),"seg")
    # print("Best solution:",best_pos)
    # print("Fitness:",best_cost)
    # print("Solución encontrada en la iteración:", optimizer.cost_history.index(best_cost))
    improvement = 100 * (optimizer.cost_history[0] - best_cost) / (optimizer.cost_history[0])
    # print(f"Improvement over greedy heuristic: {improvement: .2f}%")
    table.append([i+1,best_pos[0],best_pos[1],best_cost,optimizer.cost_history.index(best_cost),improvement,end_time])
    best_solutions.append(best_cost)
    best_iterations.append(optimizer.cost_history.index(best_cost))
    best_times.append(end_time)
    if i == 0:
        plot_cost_history(optimizer.cost_history)
        plt.show()
print(tabulate(table, headers=["Iteration", "x1","x2","Best fitness","Best iteration","Improvement","Running time"],tablefmt="outline"))
print("===============================================")
print("Costo total promedio =", statistics.mean(best_solutions))
varianza = statistics.variance(best_solutions)
print("Varianza: ", varianza)
print("Mejor solución encontrada: %f" % (min(best_solutions)))

df = pd.DataFrame(best_solutions, columns =['Fitness'], dtype = float)
df['Time'] = best_times
df['Iteracion'] = best_iterations
df.to_excel('resultado-PSO.xlsx')
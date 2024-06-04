"""
Numerical Methods Package: Parallel Utilities
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""
import matplotlib.pyplot as plt
import concurrent.futures
from Utilities import pde_convergence_analysis, calculate_dt

def run_solver(solver, name, a, b, d, initial_condition, boundary_condition_0, boundary_condition_1, alpha, nx_values, exact_solution):
    dt_values = [calculate_dt((b - a) / nx, alpha) for nx in nx_values]
    hs, errors = pde_convergence_analysis(solver, a, b, d, initial_condition, boundary_condition_0, boundary_condition_1, alpha, nx_values, dt_values, exact_solution)
    return hs, errors, name

def parallel_plot_pde_convergence(solvers, solver_names, a, b, d, initial_condition, boundary_condition_0, boundary_condition_1, alpha, nx_values, exact_solution, title):
    plt.figure(figsize=(12, 8))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_solver = {executor.submit(run_solver, solver, name, a, b, d, initial_condition, boundary_condition_0, boundary_condition_1, alpha, nx_values, exact_solution): name for solver, name in zip(solvers, solver_names)}

        for future in concurrent.futures.as_completed(future_to_solver):
            name = future_to_solver[future]
            try:
                hs, errors, solver_name = future.result()

                # Debug statements to check the data
                print(f"Solver: {solver_name}")
                print(f"hs: {hs}")
                print(f"errors: {errors}")

                # Ensure hs and errors are not empty
                if hs and errors:
                    plt.loglog(hs, errors, label=f"{solver_name}")
                else:
                    print(f"No data to plot for {solver_name}")

            except Exception as exc:
                print(f"{name} generated an exception: {exc}")

    plt.title(f"Convergence Analysis for {title}")
    plt.xlabel('Spatial Step Size (dx)')
    plt.ylabel('Error')
    plt.gca().invert_xaxis()  # Reverse the x-axis direction
    plt.legend()
    plt.grid(True, which="both", ls='--')
    plt.show()
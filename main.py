import torch
import pandas as pd
import numpy as np
from SimCIM import *
from functions import *
import matplotlib.pyplot as plt
import cProfile
import pstats

datatype = torch.float32
device = 'cpu'

def test():
    """
    Function to test the QUBO optimization process for a predefined TSP instance.

    This function defines a distance matrix, calculates the optimal tour, and
    runs the SimCIM solver to compare its performance with the optimal route.
    """
    matrix = np.empty((5, 5))
    for i in range (5):
        matrix[i][i] = 0.0
    matrix[0][1] = matrix[1][0] = 2.0
    matrix[0][2] = matrix[2][0] = 9.0
    matrix[0][3] = matrix[3][0] = 10.0
    matrix[0][4] = matrix[4][0] = 3.0
    matrix[1][2] = matrix[2][1] = 7.0
    matrix[1][3] = matrix[3][1] = 5.0
    matrix[1][4] = matrix[4][1] = 4.0
    matrix[2][3] = matrix[3][2] = 8.0
    matrix[2][4] = matrix[4][2] = 6.0
    matrix[4][3] = matrix[3][4] = 7.0

    optRoute = [0, 1, 3, 2, 4, 0]  # Predefined optimal route
    optDist = totalDist(optRoute, matrix)
    print("Optimal solution has total distance = %0.1f " % optDist)

    # Run SimCIM solver
    path, energy, iteration = run_simcim(matrix)

    # Print results
    loss = (energy - optDist) / optDist
    print("Loss %f" % loss)
    print("Convergence iteration %d, with best route %d" % (iteration, energy))

def tspMatrix(df, TSPSize):
    """
    Construct a dictionary mapping TSP solutions to their distance matrices.

    Args:
        df (pandas.DataFrame): Data containing TSP instances.
        TSPSize (int): Number of cities in the TSP instance.

    Returns:
        dict: Dictionary mapping solutions to their respective distance matrices.
    """
    matrixDict = {}
    # Each entry of the dictionary is a string of the solution and a distance matrix of the problem
    for row in df.index:
        solution = df.at[row, 'sol']
        matrix = np.empty((TSPSize, TSPSize))
        for i in range(TSPSize):
            for j in range(i, TSPSize):
                x1 = df.at[row, 'X_'+str(i)]
                y1 = df.at[row, 'Y_'+str(i)]
                x2 = df.at[row, 'X_'+str(j)]
                y2 = df.at[row, 'Y_'+str(j)]
                matrix[i][j] = matrix[j][i] = calcDistance(x1, y1, x2, y2)

        matrixDict[solution] = matrix

    return matrixDict

def calcDistance(x1, y1, x2, y2):
    """
    Compute Euclidean distance between two points.

    Args:
        x1, y1 (float): Coordinates of the first point.
        x2, y2 (float): Coordinates of the second point.

    Returns:
        float: Distance between the two points.
    """
    dist = np.sqrt(pow((x1-x2),2) + pow((y1-y2),2))
    return dist

def totalDist(route, distMatrix):
    """
    Compute the total distance of a given route using the distance matrix.

    Args:
        route (list): Order of cities visited.
        distMatrix (numpy.ndarray): Distance matrix between cities.

    Returns:
        float: Total distance of the route.
    """
    d = 0.0  # total distance between cities
    size = len(route)
    for i in range(size-1):
        d += distMatrix[int(route[i])][int(route[i+1])]
    return d

def error(route, distMatrix, optDist):
    """
    Compute the error between computed and optimal route distances.

    Args:
        route (list): Computed route sequence.
        distMatrix (numpy.ndarray): Distance matrix between cities.
        optDist (float): Optimal tour distance.

    Returns:
        float: Computed error value.
    """
    d = totalDist(route, distMatrix)
    return d - optDist

def TSP15():
    """
    Solve 20k TSP instances of 15 cities each using SimCIM.
    """
    print("\nBegin TSP using quantum inspired annealing - TSP15")
    TSPSize = 15
    data = pd.read_csv('tspData.csv', dtype={'sol': 'string'})

    # Calculate a distance matrix for every TSP instance
    tspDict = tspMatrix(data, TSPSize)
    numCities = TSPSize
    loss_values = []
    convergence = []
    n = 0
    sum_iteration = sum_loss = 0
    error = 0

    for r in tspDict:
        print("Setting num_cities = %d " % numCities)
        print("Optimal solution is " + r)
        n += 1
        print("Tour number %d" % n)
        # Convert optimal solution to a list
        optRoute = r.split("-")
        # Calculate distance of optimal route
        optDist = totalDist(optRoute, tspDict[r])
        print("Optimal solution has total distance = %0.1f " % optDist)

        # Create a distance matrix for the problem
        distance_matrix = torch.from_numpy(tspDict[r]).float()

        # Minimise the problem
        path, energy, iteration = run_simcim(distance_matrix.numpy())

        # Check that the resulting path is correct
        if path.shape[0] == distance_matrix.shape[0]:

            # Print performance metrics
            loss = (energy - optDist)/optDist
            loss_values.append(loss)
            convergence.append(iteration)
            sum_iteration += iteration
            sum_loss += loss
            print("Loss %f" % loss)
            print("Convergence iteration %d, with best route %d" % (iteration, energy))
            print("Average iteration %d, average loss %f" % ((sum_iteration / n), (sum_loss / n)))

        else:
            print("ERROR! Invalid path")
            error += 1

        print("Error rate %f" % (error/n))

    # x_axis = range(0, n)
    # plt.title("Precision")
    # plt.plot(x_axis, loss_values)
    # plt.show()
    #
    # x_axis = range(0, n)
    # plt.title("Convergence")
    # plt.plot(x_axis, convergence)
    # plt.show()

def national_TSP():
    """
    Solve real-world TSP instances using SimCIM.
    """
    print("\nBegin TSP using quantum inspired annealing - National TSP")

    """Read a National TSP file and return a list of coordinates."""
    data = [(27603, "wi29.tsp"),(6656, "dj38.tsp"),(9352, "qa194.tsp"),(79114, "uy734.tsp"),(95345, "zi929.tsp"),
            (11340, "lu980.tsp"),(26051, "rw1621.tsp"),(86891, "mu1979.tsp"),(96132, "nu3496.tsp"),(1290319, "ca4663.tsp"),
            (394543, "tz6117.tsp"),(172350, "eg7146.tsp"),(238242, "ym7663.tsp"),(114831, "pm8079.tsp"),(206128, "ei8246.tsp"),
            (837377, "ar9152.tsp"),(491869, "ja9847.tsp"),(300876, "gr9882.tsp"),(1061387, "kz9976.tsp"),(520383, "fi10639.tsp"),
            (427246, "mo14185.tsp"),(176940, "ho14473.tsp"),(557274, "it16862.tsp"),(569115, "vm22775.tsp"),(855331,"sw24978.tsp"),
            (959011, "bm33708.tsp"),(4565452, "ch71009.tsp")]
    loss_values = []
    convergence = []
    n = 0
    sum_iteration = sum_loss = 0
    error = 0

    # Optimise each TSP problem
    for file in range(len(data)):
        n += 1
        optDist = data[file][0]
        print("place %d, optimal tour %d" % (n, optDist))
        with open(data[file][1], 'r') as file:
            lines = file.readlines()

        # Extract the coordinates of the problem
        coordinates = []
        for line in lines:
            parts = line.strip().split()
            if parts[0].isdigit():  # This checks if the line starts with a node number
                x, y = float(parts[1]), float(parts[2])
                coordinates.append((x, y))
        TSPSize = len(coordinates)
        print("TSP size %d" % TSPSize)

        # Iterating through the points two at a time
        matrix = np.empty((TSPSize, TSPSize))
        total_distance = 0
        count = 0

        # Create a distance matrix for the problem
        for i in range(TSPSize):
            for j in range(i, TSPSize):
                matrix[i][j] = matrix[j][i] = round(calcDistance(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1]))
                # total_distance += matrix[i][j] if i != j else 0
                # count += 1 if i != j else 0

        # Calculate the average distance
        # average_distance = total_distance / count
        # Normalize the matrix by the average distance
        # matrix /= average_distance
        # matrix *= 100

        # Minimise the problem
        path, energy, iteration = run_simcim(matrix)

        # Check that the resulting path is correct
        if path.shape[0] == matrix.shape[0]:

            # Print performance metrics
            loss = (energy - optDist) / optDist
            loss_values.append(loss)
            convergence.append(iteration)
            sum_iteration += iteration
            sum_loss += loss
            print("Loss %f" % loss)
            print("Convergence iteration %d, with best route %d" % (iteration, energy))
            print("Average iteration %d, average loss %f" % ((sum_iteration / n), (sum_loss / n)))

        else:
            print("ERROR! Invalid path")
            error += 1

        print("Error rate %f" % (error / n))

def TSPLIB():
    """
    Solve TSP benchmarking instances using SimCIM.
    """
    print("\nBegin TSP using quantum inspired annealing - TSPLIB")

    # Read a TSPLIB file and return a list of coordinates
    data = [("ulysses16"), ("ulysses22"), ("gr24"), ("fri26"), ("bayg29"), ("bays29"), ("att48"), ("gr48"), ("eil51"),
            ("berlin52"),("st70"), ("eil76"), ("pr76"), ("gr96"), ("rd100"), ("kroA100"), ("kroC100"), ("kroD100"),
            ("eil101"),("lin105"), ("gr120"),("ch130"), ("ch150"), ("brg180"), ("gr202"), ("tsp225"), ("a280"),
            ("pcb442"), ("pa561"), ("gr666")]

    loss_values = []
    convergence = []
    n = 0
    sum_iteration = sum_loss = 0

    # Optimise each TSP problem
    for file in range(len(data)):
        n += 1
        dataFile = data + ".tsp"
        with open(dataFile, 'r') as file:
            lines = file.readlines()

        # Extract coordinates
        coordinates = []
        for line in lines:
            parts = line.strip().split()
            if parts[0].isdigit():  # Check the line starts with a node number
                x, y = float(parts[1]), float(parts[2])
                coordinates.append((x, y))
        TSPSize = len(coordinates)
        print("TSP size %d" % TSPSize)

        # Iterating through the points two at a time and create a distance matrix for the problem
        matrix = np.empty((TSPSize, TSPSize))
        for i in range(TSPSize):
            for j in range(i, TSPSize):
                matrix[i][j] = matrix[j][i] = round(
                    calcDistance(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1]))

        optTourFile = data + ".opt.tour"
        with open(optTourFile, 'r') as file:
            lines = file.readlines()

        optDist = totalDist(lines, matrix)
        print("place %d, optimal tour %d" % (n, optDist))

        # Minimise the problem
        path, energy, iteration = run_simcim(matrix)

        # Print performance metrics
        loss = (energy - optDist) / optDist
        loss_values.append(loss)
        convergence.append(iteration)
        sum_iteration += iteration
        sum_loss += loss
        print("Loss %f" % loss)
        print("Convergence iteration %d, with best route %d" % (iteration, energy))
        print("Average iteration %d, average loss %f" % ((sum_iteration / n), (sum_loss / n)))

def run_simcim(lengths):
    """
    Run the SimCIM solver on a given distance matrix for the TSP problem.

    Args:
        lengths (numpy.ndarray): Distance matrix between cities.

    Returns:
        tuple: (optimal order, best route distance, number of iterations)
    """
    N_cities = lengths.shape[0]

    B = 0.1  # Default penalty coefficient for path constraints
    A = 1.5 * B * lengths.max()  # Default penalty coefficient for city constraints
    J, b = get_Jh(lengths, A, B)

    simcim = Simcim(J, b, device, datatype)

    # Profiler stats
    profiler = cProfile.Profile()
    profiler.enable()

    # Optimise the problem
    c_current, c_evol, iteration = simcim.evolve()

    profiler.disable()

    # Print profiling statistics
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(10)

    s_cur = torch.sign(c_current)
    E = energy(J, b, s_cur)

    print('Evolution of amplitudes')
    fig, ax = plt.subplots()#(figsize=(5, 5))
    plt.title("Evolution of amplitudes - TSP")
    for i in range(J.size(0)):
        ax.plot(c_evol[i, :].cpu().numpy())
    #ax.grid()
    ax.set_xlabel('iterations')
    ax.set_ylabel('amplitudes')
    fig.tight_layout()
    fig.show()

    s_min = s_cur[:, torch.argmin(E)]
    order = get_order_simcim(s_min, N_cities)
    print(order)
    distance = totalDist(order, lengths) + lengths[order[0]][order[(order.shape[0])-1]]
    #fig = plot_cities(cities, lengths, order)
    print('Best route among ' + str(simcim.params_disc['attempt_num']) + ' runs')
    #fig.show()

    return order, distance, int(iteration)


def main():
    """
    Entry point for solving the TSP problem using SimCIM.
    """
    national_TSP()

if __name__ == "__main__":
    main()
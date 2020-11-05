# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
import time

import build.pydiffmem.core as dm
import pandas as pd
import numpy as np

if __name__ == '__main__':

    X = pd.read_csv("example_data.csv", header=0, names=["Id", "Dose", "Time", "Concentration", "Weight", "Sex"])
    A = np.diag([1., 1., 1.])
    B = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.double, order='F')

    population = dm.Population()
    for name, group in X.groupby('Id'):
        id_ = int(name)
        t = group['Time'].to_numpy()
        amt = group['Dose'].to_numpy()
        y = group['Concentration'].to_numpy()
        y = np.reshape(y, (1, -1), order='F')
        sm = dm.StructuralModel("PK", {"time": t, "amt": amt})
        lm = dm.LikelihoodModel("Gaussian", {'t': t, 'y': y})
        population.add(dm.Individual(id_, sm, lm, A=A, B=B))

    beta = np.array([.7, .5, -1.], np.double, order='F')

    estimated = np.array([True, True, True], np.uint8, order='F')
    omega = np.eye(2, dtype=np.double)
    cov_model = np.array([[True, True], [True, True]], np.uint8, order='F')
    sigma = np.array([1.], np.double, order='F')
    sigma_model = np.array([True], np.uint8, order='F')

    model = dm.MEModel(population, beta=beta, estimated=estimated, omega=omega, cov_model=cov_model, sigma=sigma,
                            sigma_model=sigma_model)
    print("===============trying to perform SAEM=================")
    time_start = time.time()
    dm.saem(model, k1=120, k2=20)
    time_end = time.time()
    print(f"==============SAEM took {time_end-time_start}==============")


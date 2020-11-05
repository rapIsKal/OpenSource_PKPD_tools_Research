import pandas as pd
import numpy as np
import pydiffmem.core as dm
import unittest
from utilities import CustomTestCase

class TestMEModel(CustomTestCase):
    def setUp(self):
        X = pd.read_csv("simple.csv", header=0, names=["id", "t", "yr", "y"])
        A = np.diag([1.,1.,1.])
        B = np.array([[0,0], [1, 0], [0, 1]], dtype=np.double, order='F')

        population = dm.Population()
        for name, group in X.groupby('id'):
            id_ = int(name)
            t = group['t'].as_matrix()
            y = group['y'].as_matrix()
            y = np.reshape(y, (1, -1), order='F')
            sm = dm.StructuralModel("Linear")
            lm = dm.LikelihoodModel("Gaussian", {'t':t, 'y':y})
            population.add( dm.Individual(id_, sm, lm, A=A, B=B) )
        self.population = population

        beta = np.array([.7, .5, -1.], np.double, order='F')

        estimated = np.array([True, True, True], np.uint8, order='F')
        omega = np.eye(2, dtype=np.double)
        cov_model = np.array([ [True, True], [True, True] ], np.uint8, order='F')
        sigma = np.array([1.], np.double, order='F')
        sigma_model = np.array([True], np.uint8, order='F')

        self.model = dm.MEModel(population, beta=beta, estimated=estimated, omega=omega, cov_model=cov_model, sigma=sigma, sigma_model=sigma_model)

    def test_model(self):
        self.assertEqual(self.model.numberOfRandom(), 2)
        self.assertEqual(self.model.numberOfBetas(), 3)
        self.assertEqual(self.model.numberOfTaus(), 1)

    def test_saem(self):
        model = self.model
        dm.saem(model, k1=120, k2=20)

        beta_true = np.array([0.862853, 0.360712, -1.27187]).reshape((3,1))
        omega_true = np.array([[0.9, 0.5], [0.5, 1.0]])
        sigma_true = np.array([.1]).reshape((1,1))

        self.npAssertAlmostEqual(model.beta(), beta_true, atol=1e-1)
        self.npAssertAlmostEqual(model.omega(), omega_true, atol=1e-1)
        self.npAssertAlmostEqual(model.sigma(), sigma_true, atol=1e-1)


if __name__ == '__main__':
    unittest.main()

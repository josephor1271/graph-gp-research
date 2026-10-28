import unittest

import gpytorch
import numpy as np
import scipy.linalg
import torch

from chebyshev import apply_chebyshev, heat_coefficients
from demo import grid_laplacian
from dense_gp import DenseHeatKernel, GraphGP


class GraphGPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.laplacian = grid_laplacian(3)
        self.dense_laplacian = self.laplacian.to_dense()
        self.ids = torch.arange(9)

    def test_dense_covariance_and_gradient(self):
        kernel = DenseHeatKernel(self.laplacian, heat_time=1.3)
        actual = kernel(self.ids).to_dense()
        reference = torch.matrix_exp(-kernel.heat_time * self.dense_laplacian)
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-11)
        weights = torch.randn(9, 9, dtype=torch.float64)
        actual_gradient, = torch.autograd.grad((actual * weights).sum(), kernel.raw_heat_time)
        reference_gradient, = torch.autograd.grad((reference * weights).sum(), kernel.raw_heat_time)
        torch.testing.assert_close(actual_gradient, reference_gradient, atol=1e-12, rtol=1e-10)

    def test_dense_posterior_matches_conditioning(self):
        train = torch.tensor([0, 2, 3, 7])
        test = torch.tensor([1, 4, 8])
        y = torch.tensor([0.2, -0.4, 0.1, 0.6], dtype=torch.float64)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        likelihood.noise = 0.04
        model = GraphGP(train, y, likelihood, DenseHeatKernel(self.laplacian, 1.3))
        model.eval()
        with torch.no_grad():
            prediction = model(test)
            covariance = torch.matrix_exp(-1.3 * self.dense_laplacian)
            observed = covariance[train][:, train] + likelihood.noise * torch.eye(4, dtype=torch.float64)
            cross = covariance[test][:, train]
            mean = cross @ torch.linalg.solve(observed, y)
            posterior = covariance[test][:, test] - cross @ torch.linalg.solve(observed, cross.T)
        torch.testing.assert_close(prediction.mean, mean, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(prediction.covariance_matrix, posterior, atol=1e-12, rtol=1e-10)

    def test_chebyshev_action_and_heat_derivative(self):
        rhs = torch.randn(9, 3, dtype=torch.float64)
        weights = torch.randn_like(rhs)
        for time in [0.05, 0.8, 2.85602691, 4.0]:
            with self.subTest(heat_time=time):
                tau = torch.tensor(time, dtype=torch.float64, requires_grad=True)
                coefficients = heat_coefficients(tau)
                actual = apply_chebyshev(self.laplacian, coefficients, rhs)
                # SciPy's matrix exponential and analytic derivative share no
                # coefficient, recurrence, or eigendecomposition implementation.
                exact = torch.from_numpy(scipy.linalg.expm(-time * self.dense_laplacian.numpy()))
                reference = exact @ rhs
                derivative = (-self.dense_laplacian @ exact @ rhs * weights).sum()
                actual_derivative, = torch.autograd.grad((actual * weights).sum(), tau)
                torch.testing.assert_close(actual, reference, atol=2e-12, rtol=2e-11)
                torch.testing.assert_close(actual_derivative, derivative, atol=2e-12, rtol=2e-10)
                vector = apply_chebyshev(self.laplacian, coefficients, rhs[:, 0])
                torch.testing.assert_close(vector, reference[:, 0], atol=2e-12, rtol=2e-11)

    def test_polynomial_on_declared_interval(self):
        spectrum = np.linspace(0, 2, 2001)
        worst_error = 0.0
        minimum = float("inf")
        for time in np.linspace(0.05, 4, 17):
            coefficients = heat_coefficients(torch.tensor(time, dtype=torch.float64)).numpy()
            values = np.polynomial.chebyshev.chebval(spectrum - 1, coefficients)
            worst_error = max(worst_error, np.max(np.abs(values - np.exp(-time * spectrum))))
            minimum = min(minimum, values.min())
        self.assertLess(worst_error, 2e-12)
        self.assertGreater(minimum, 0)
        print(f"\nDegree 20 sampled spectral error: {worst_error:.3e}; minimum: {minimum:.3e}")

    def test_constant_and_linear_recurrence(self):
        rhs = torch.randn(9, 2, dtype=torch.float64)
        constant = torch.tensor([1.7], dtype=torch.float64)
        linear = torch.tensor([1.7, -0.2], dtype=torch.float64)
        torch.testing.assert_close(apply_chebyshev(self.laplacian, constant, rhs), 1.7 * rhs)
        reference = 1.7 * rhs - 0.2 * (self.dense_laplacian @ rhs - rhs)
        torch.testing.assert_close(apply_chebyshev(self.laplacian, linear, rhs), reference)


if __name__ == "__main__":
    unittest.main()

import unittest

import gpytorch
import torch

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


if __name__ == "__main__":
    unittest.main()

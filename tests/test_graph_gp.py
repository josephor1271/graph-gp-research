import unittest
from unittest.mock import patch

import gpytorch
import numpy as np
import scipy.linalg
import torch

from chebyshev import apply_chebyshev, heat_coefficients
from demo import grid_laplacian
from dense_gp import DenseHeatKernel, GraphGP
from graph_gp import HeatOperator, SparseHeatKernel


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
        likelihood.noise = torch.tensor(0.04, dtype=torch.float64)
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

    def test_sparse_subsets_diagonal_and_matmul_gradient(self):
        kernel = SparseHeatKernel(self.laplacian, heat_time=1.3)
        rows = torch.tensor([8, 1, 1, 4])
        columns = torch.tensor([0, 6, 2])
        operator = kernel(rows, columns).evaluate_kernel()
        exact = torch.matrix_exp(-kernel.heat_time * self.dense_laplacian)
        rhs = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        actual = operator @ rhs
        reference = exact[rows][:, columns] @ rhs
        torch.testing.assert_close(actual, reference, atol=2e-12, rtol=2e-11)
        actual_gradients = torch.autograd.grad(actual.square().sum(), (kernel.raw_heat_time, rhs))
        reference_gradients = torch.autograd.grad(reference.square().sum(), (kernel.raw_heat_time, rhs))
        for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients):
            torch.testing.assert_close(actual_gradient, reference_gradient, atol=2e-12, rtol=2e-10)
        torch.testing.assert_close(operator.mT.to_dense(), exact.detach()[columns][:, rows], atol=2e-12, rtol=2e-11)
        torch.testing.assert_close(kernel(rows, diag=True), exact.detach().diagonal()[rows], atol=2e-12, rtol=2e-11)
        torch.testing.assert_close(operator[torch.tensor([0, 3]), torch.tensor([1, 2])],
                                   exact.detach()[rows[[0, 3]], columns[[1, 2]]], atol=2e-12, rtol=2e-11)

    def test_same_model_mll_gradient_and_iterative_prediction(self):
        train = torch.tensor([0, 2, 3, 5, 6, 8])
        test = torch.tensor([1, 4, 7])
        y = torch.randn(6, dtype=torch.float64)
        models = []
        exact_values = []
        iterative_values = []
        for kernel_type in [DenseHeatKernel, SparseHeatKernel]:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
            likelihood.noise = torch.tensor(0.04, dtype=torch.float64)
            kernel = kernel_type(self.laplacian, heat_time=1.3)
            model = GraphGP(train, y, likelihood, kernel)
            models.append(model)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            value = mll(model(train), y)
            gradient, = torch.autograd.grad(value, kernel.raw_heat_time)
            exact_values.append(torch.stack([value.detach(), gradient]))
            torch.manual_seed(71)
            with (
                gpytorch.settings.max_cholesky_size(0),
                gpytorch.settings.max_preconditioner_size(0),
                gpytorch.settings.cg_tolerance(1e-10),
                gpytorch.settings.num_trace_samples(16),
                gpytorch.settings.max_lanczos_quadrature_iterations(6),
            ):
                value = mll(model(train), y)
                gradient, = torch.autograd.grad(value, kernel.raw_heat_time)
                iterative_values.append(torch.stack([value.detach(), gradient]))
        torch.testing.assert_close(exact_values[0], exact_values[1], atol=2e-11, rtol=2e-10)
        # Common probes compare representations without asking a noisy gradient
        # to equal the exact trace. The independent check below is deterministic.
        torch.testing.assert_close(iterative_values[0], iterative_values[1], atol=2e-9, rtol=2e-8)

        raw = models[1].covar_module.raw_heat_time.item()
        constraint = models[1].covar_module.raw_heat_time_constraint
        lower = constraint.lower_bound.item()
        upper = constraint.upper_bound.item()

        def independent_mll(raw_time):
            tau = lower + (upper - lower) / (1 + np.exp(-raw_time))
            covariance = scipy.linalg.expm(-tau * self.dense_laplacian.numpy())
            observed = covariance[np.ix_(train.numpy(), train.numpy())] + 0.04 * np.eye(6)
            quadratic = y.numpy() @ np.linalg.solve(observed, y.numpy())
            return -0.5 * (quadratic + np.linalg.slogdet(observed)[1] + 6 * np.log(2 * np.pi)) / 6

        step = 1e-5
        reference_gradient = (independent_mll(raw + step) - independent_mll(raw - step)) / (2 * step)
        self.assertAlmostEqual(exact_values[1][0].item(), independent_mll(raw), places=11)
        self.assertAlmostEqual(exact_values[1][1].item(), reference_gradient, places=8)

        for model in models:
            model.eval()
        with torch.no_grad():
            dense = models[0](test)
            dense_mean = dense.mean
            dense_variance = dense.variance
        with (
            torch.no_grad(),
            gpytorch.settings.max_cholesky_size(0),
            gpytorch.settings.max_preconditioner_size(0),
            gpytorch.settings.max_eager_kernel_size(0),
            gpytorch.settings.eval_cg_tolerance(1e-10),
        ):
            sparse_prediction = models[1](test)
            torch.testing.assert_close(sparse_prediction.mean, dense_mean, atol=2e-10, rtol=2e-9)
            torch.testing.assert_close(sparse_prediction.variance, dense_variance, atol=2e-10, rtol=2e-9)

    def test_sparse_inference_uses_products_without_full_graph_covariance(self):
        train = torch.tensor([0, 2, 3, 5, 6, 8])
        test = torch.tensor([1, 4, 7])
        y = torch.randn(6, dtype=torch.float64)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        likelihood.noise = torch.tensor(0.04, dtype=torch.float64)
        kernel = SparseHeatKernel(self.laplacian, heat_time=1.3)
        model = GraphGP(train, y, likelihood, kernel)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        product_shapes = []
        original_to_dense = HeatOperator.to_dense
        original_tensor_to_dense = torch.Tensor.to_dense

        def guarded_to_dense(operator):
            self.assertNotEqual(tuple(operator.shape), (9, 9))
            return original_to_dense(operator)

        def guarded_tensor_to_dense(tensor, *args, **kwargs):
            self.assertEqual(tensor.layout, torch.strided, "Sparse graph was densified")
            return original_tensor_to_dense(tensor, *args, **kwargs)

        def tracked_action(laplacian, coefficients, rhs):
            self.assertEqual(laplacian.layout, torch.sparse_coo)
            self.assertNotEqual(tuple(rhs.shape), (9, 9), "Full graph basis was constructed")
            product_shapes.append(tuple(rhs.shape))
            return apply_chebyshev(laplacian, coefficients, rhs)

        with (
            patch.object(HeatOperator, "to_dense", guarded_to_dense),
            patch.object(torch.Tensor, "to_dense", guarded_tensor_to_dense),
            patch("graph_gp.apply_chebyshev", side_effect=tracked_action),
            gpytorch.settings.max_cholesky_size(0),
            gpytorch.settings.max_preconditioner_size(0),
            gpytorch.settings.max_eager_kernel_size(0),
            gpytorch.settings.cg_tolerance(1e-10),
            gpytorch.settings.eval_cg_tolerance(1e-10),
            gpytorch.settings.num_trace_samples(16),
            gpytorch.settings.max_lanczos_quadrature_iterations(6),
        ):
            loss = -mll(model(train), y)
            loss.backward()
            self.assertTrue(torch.isfinite(kernel.raw_heat_time.grad))
            model.eval()
            with torch.no_grad():
                posterior = model(test)
                self.assertTrue(torch.isfinite(posterior.mean).all())
                self.assertTrue(torch.isfinite(posterior.variance).all())
        self.assertTrue(product_shapes)
        self.assertTrue(any(columns > 1 for _, columns in product_shapes))
        print(f"Sparse training/backward/prediction: {len(product_shapes)} recurrence calls; no full graph covariance")


if __name__ == "__main__":
    unittest.main()

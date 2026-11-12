import unittest

import gpytorch
import torch

from dense_gp import GraphGP
from learned_polynomial import LearnedPolynomialKernel


def check_kernel(kernel):
    """Dense matrix-power reference, also used on an actual traffic slice."""
    base = kernel.base_kernel
    L = base.laplacian.to_dense()
    n = len(L)
    identity = torch.eye(n, dtype=L.dtype)
    train, test = torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5])
    y = torch.tensor([.2, -.7, .4], dtype=L.dtype)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=L.dtype)
    model = GraphGP(train, y, likelihood, kernel)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    rows, columns = torch.tensor([5, 0, 2, 2]), torch.tensor([4, 1, 0])
    diagnostics = []
    for values in (None, (.3, -.2), (-.9, .2)):
        model.train()
        if values is not None:
            base.initialize(a1=values[0], a2=values[1])
        Q = identity + base.a1 * L + base.a2 * (2 * (L @ L) - 4 * L)
        if values is None:
            beta = 2 * torch.sigmoid(L.new_tensor(1.))
            torch.testing.assert_close(Q, identity-beta*L/2+beta.square()*(L@L)/8)
        # q(0)=c0-c1+c2=1, even when the filter changes sign elsewhere.
        torch.testing.assert_close(1+base.a1-base.a2-base.a1+base.a2, L.new_tensor(1.))
        reference = kernel.outputscale * (Q @ Q.T)
        actual = kernel(torch.arange(n)).to_dense()
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(actual, actual.T, atol=1e-12, rtol=1e-10)
        assert torch.linalg.eigvalsh(actual.detach()).min() > -1e-10
        operator = kernel(rows, columns).evaluate_kernel()
        rhs = torch.arange(6, dtype=L.dtype).reshape(3, 2)/6
        torch.testing.assert_close(operator @ rhs, reference[rows][:, columns] @ rhs, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(operator.mT.to_dense(), reference[columns][:, rows], atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(kernel(rows, diag=True), reference.diagonal()[rows], atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(operator[:2, :2].to_dense(), reference[rows[:2]][:, columns[:2]], atol=1e-10, rtol=1e-9)
        H = reference[train][:, train] + likelihood.noise * torch.eye(len(train), dtype=L.dtype)
        factor = torch.linalg.cholesky(H)
        expected = (y @ torch.linalg.solve(H, y) + 2*factor.diagonal().log().sum() + len(train)*torch.log(L.new_tensor(2*torch.pi)))/(2*len(train))
        loss = -mll(model(train), y)
        torch.testing.assert_close(loss, expected, atol=1e-10, rtol=1e-9)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        dense_gradients = torch.autograd.grad(expected, tuple(model.parameters()), retain_graph=True)
        for a, b in zip(gradients, dense_gradients):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)
        model.eval()
        with torch.no_grad():
            prediction = model(test)
            cross = reference[test][:, train]
            mean = cross @ torch.linalg.solve(H, y)
            variance = reference.diagonal()[test] - (cross * torch.linalg.solve(H, cross.T).T).sum(1)
            assert (variance > 0).all()
            torch.testing.assert_close(prediction.mean, mean, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(prediction.variance, variance, atol=1e-10, rtol=1e-9)
            target = y.flip(0)
            nlpd = -torch.distributions.Normal(prediction.mean, (prediction.variance+likelihood.noise).sqrt()).log_prob(target).mean()
            dense_nlpd = .5*(torch.log(2*torch.pi*(variance+likelihood.noise))+(target-mean).square()/(variance+likelihood.noise)).mean()
            torch.testing.assert_close(nlpd, dense_nlpd, atol=1e-10, rtol=1e-9)
        diagnostics.append(dict(covariance_error=float((actual-reference).detach().abs().max()),
            nll_error=float((loss-expected).detach().abs()), gradient_error=max(float((a-b).abs().max()) for a,b in zip(gradients,dense_gradients))))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
    optimizer.zero_grad()
    (-mll(model(train), y)).backward()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return diagnostics


class LearnedPolynomialTests(unittest.TestCase):
    def test_weighted_graph(self):
        weights = torch.diag(torch.tensor([1., 2., .7, 1.3, .8], dtype=torch.float64), 1)
        weights = weights + weights.T
        weights[0, 3] = weights[3, 0] = .4
        d = weights.sum(1).rsqrt()
        L = torch.eye(6, dtype=torch.float64) - d[:, None]*weights*d[None, :]
        kernel = gpytorch.kernels.ScaleKernel(LearnedPolynomialKernel(L.to_sparse_coo())).double()
        with gpytorch.settings.max_cholesky_size(800):
            check_kernel(kernel)

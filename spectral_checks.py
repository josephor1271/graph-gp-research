"""Independent, six-vertex checks for the frozen spectral screen."""
import gpytorch
import torch
from experiment_spectral_expressivity import Poly4, heat_block, model_for, posterior

def check():
    errors = {}
    def compare(key, actual, reference, gradient=False):
        torch.testing.assert_close(actual, reference, atol=1e-9 if gradient else 1e-10, rtol=1e-8 if gradient else 1e-9)
        errors[key] = max(errors.get(key, 0.), (actual-reference).detach().abs().max().item())

    weights = torch.diag(torch.tensor([1., 2., .7, 1.3, .8]), 1)
    weights = weights + weights.T
    weights[0, 3] = weights[3, 0] = .4
    d = weights.sum(1).rsqrt()
    eye = torch.eye(6)
    graph = eye-d[:, None]*weights*d
    eig, U = torch.linalg.eigh(graph)
    compare("eigenbasis", (U*eig) @ U.T, graph)
    assert eig.min() >= -1e-10 and eig.max() <= 2+1e-10
    train, test, y = torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5]), torch.tensor([.2, -.7, .4])
    ids = torch.cat((train, test))
    model = model_for("Poly", graph, None, train, y)
    kernel = model.covar_module
    for state in ("degree2", "nested", "sign-changing"):
        model.train()
        if state == "nested":
            kernel.base_kernel = Poly4(graph.to_sparse())
        base = kernel.base_kernel
        if state == "sign-changing":
            base.initialize(a1=-.9, a2=.2, a3=.15, a4=-.12)
        x = eig-1
        q = 1+base.a1*(x+1)+base.a2*(2*x.square()-2)
        if state != "degree2":
            q = q+base.a3*(4*x**3-3*x+1)+base.a4*(8*x**4-8*x.square())
        compare("normalization", q[0], eig.new_tensor(1.))
        assert state != "sign-changing" or q.min() < 0 < q.max()
        ref = (U*(kernel.outputscale*q.square())) @ U.T
        actual = kernel(torch.arange(6)).to_dense()
        compare("covariance", actual, ref)
        compare("symmetry", actual, actual.T)
        assert torch.linalg.eigvalsh(actual.detach()).min() >= -1e-10
        rows, rhs = torch.tensor([5, 2, 2, 0]), torch.arange(6.).reshape(3, 2)
        op = kernel(rows, train).evaluate_kernel()
        compare("product", op @ rhs, ref[rows][:, train] @ rhs)
        compare("transpose", op.mT.to_dense(), ref[train][:, rows])
        expected, mean, _ = posterior(ref[ids][:, ids], y, model.likelihood.noise)
        loss = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(model(train), y)
        compare("nll", loss, expected)
        for a, b in zip(*(torch.autograd.grad(v, tuple(model.parameters())) for v in (loss, expected))):
            compare("raw_gradient", a, b, True)
        model.eval()
        compare("mean", model(test).mean, mean)
    old = model_for("Cheb", graph, None, train, y)
    beta0 = old.covar_module.base_kernel.heat_time
    for beta in (beta0.item(), 8., 32., 64.):
        physical = torch.stack((beta0, old.covar_module.outputscale, old.likelihood.noise.squeeze()))
        logp = torch.tensor([beta, *physical[1:].detach().tolist()]).log().requires_grad_()
        b, v, n = logp.exp()
        cov, noise = heat_block(logp, eig, U)
        direct = v*torch.matrix_exp(-b*graph)
        compare("heat_cov", cov, direct)
        loss, ref = posterior(cov, y, noise)[0], posterior(direct, y, n)[0]
        compare("heat_nll", loss, ref)
        compare("heat_log_gradient", torch.autograd.grad(loss, logp)[0], torch.autograd.grad(ref, logp)[0], True)
        if beta == beta0.item():
            compare("old_heat_cov", old.covar_module(torch.arange(6)).to_dense(), cov)
            old_loss = posterior(old.covar_module(ids).to_dense(), y, physical[2])[0]
            exact = posterior(heat_block(physical.log(), eig, U[ids])[0], y, physical[2])[0]
            compare("old_heat_nll", old_loss, exact)
            gradients = [torch.autograd.grad(z, tuple(old.parameters()), retain_graph=True) for z in (old_loss, exact, physical.sum())]
            for a, b, slope in zip(*gradients):
                compare("old_heat_physical_gradient", a/slope, b/slope, True)
    print("Numerical gate maxima", errors, flush=True)

import copy

import numpy as np

from torchy import sequential
from torchy.layer import Layer, NnLayer
from torchy.loss import CrossEntropyLoss
from torchy.optim import SGD


class GradientCheck:

    @staticmethod
    def check_gradient(f, x, delta=1e-5, tol=1e-4):
        assert isinstance(x, np.ndarray)
        assert x.dtype == np.float64

        fx, analytic_grad = f(x)
        analytic_grad = analytic_grad.copy()

        assert analytic_grad.shape == x.shape

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            analytic_grad_at_ix = analytic_grad[ix]

            x0 = copy.deepcopy(x)
            x1 = copy.deepcopy(x)

            x0[ix] += delta
            x1[ix] -= delta

            numeric_grad_at_ix = (f(x0)[0] - f(x1)[0]) / (2 * delta)

            if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
                print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (
                    ix, analytic_grad_at_ix, numeric_grad_at_ix))
                return False

            it.iternext()

        print("Gradient check passed")

        return True

    @staticmethod
    def check_layer_gradient(layer: Layer, x, delta=1e-5, tol=1e-4):
        output = layer.forward(x)
        output_weight = np.random.randn(*output.shape)

        def helper_func(x):
            output = layer.forward(x)
            loss = np.sum(output * output_weight)
            d_out = np.ones_like(output) * output_weight
            grad = layer.backward(d_out)
            return loss, grad

        return GradientCheck.check_gradient(helper_func, x, delta, tol)

    @staticmethod
    def check_layer_param_gradient(layer: NnLayer, X, param_name, delta=1e-5, tol=1e-4):
        param = layer.params()[param_name]
        initial_w = param.data

        output = layer.forward(X)
        output_w = np.random.randn(*output.shape)

        def helper(w):
            param.data = w
            output = layer.forward(X)
            loss = np.sum(output * output_w)
            d_out = np.ones_like(output) * output_w
            layer.backward(d_out)
            grad = param.grad

            return loss, grad

        return GradientCheck.check_gradient(helper, initial_w, delta, tol)

    @staticmethod
    def check_model_gradient(model: sequential.Sequential, X, y,
                             delta=1e-5, tol=1e-4):

        params = model.params()

        optimizer = SGD(params)

        for param_key in params:
            print("Checking gradient for %s" % param_key)
            param = params[param_key]
            initial_w = param.data
            criterion = CrossEntropyLoss()

            def helper_func(w):
                param.data = w

                optimizer.zero_grad()

                preds = model(X)
                loss, loss_grad = criterion(preds, y)

                model.backward(loss_grad)
                grad = copy.deepcopy(param.grad)
                return loss, grad

            if not GradientCheck.check_gradient(helper_func, initial_w, delta, tol):
                return False

        return True

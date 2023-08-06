import numpy as np

from layer import Layer, NnLayer


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

            x0 = x.copy()
            x1 = x.copy()

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


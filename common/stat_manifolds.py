import jax.numpy as jnp
from math import prod
from manifolds import *

class StatisticalManifold(Manifold):
    def __init__(self, dim, pdf, shapes):
        super().__init__(dim = dim)
        self.pdf = pdf
        self.shapes = shapes

    def coerce_parameters(self):
        """
        The parameters are a vector, but will often have more nontrivial shapes.

        This function coerces the vector into the desired shapes, which are passed as comma-separated tuples.
        """
        chunksizes = [prod(shape) for shape in self.shapes]
        param_list = jnp.split(self.parameters,
                               jnp.cumsum(chunksizes)[:-1])
        return [param.reshape(shape) for param, shape in zip(param_list, self.shapes)]

class AlphaConnection(Connection):
    def __init__(self, manifold):
        self.manifold = manifold

def multivariate_gaussian_pdf(mu_params, sigma_params):
    K = 1 / jnp.sqrt(jnp.linalg.det(2 * jnp.pi * sigma_params))
    return lambda x: K * jnp.exp(-0.5 * (x - mu_params).T @ jnp.linalg.inv(sigma_params) @ (x - mu_params))

if __name__ == "__main__":
    dim = 3

    manifold = StatisticalManifold(dim + dim**2,
                                   multivariate_gaussian_pdf,
                                   [(dim,), (dim, dim)])

from math import prod
import jax.numpy as jnp
from jax.random import PRNGKey, normal, chisquare
from jax import grad, jacrev
from manifolds import *

class StatisticalManifold(Manifold):
    def __init__(self, dim, param_dim, logpdf, shapes):
        super().__init__(dim = dim)
        self.param_dim = param_dim
        self.logpdf = logpdf
        self.shapes = shapes

    def coerce_parameters(self, parameters):
        """
        The parameters are a vector, but will often have more nontrivial shapes.

        This function coerces the vector into the desired shapes, which are passed as comma-separated tuples.
        """
        chunksizes = jnp.array([prod(shape) for shape in self.shapes])
        param_list = jnp.split(parameters,
                               jnp.cumsum(chunksizes)[:-1])
        return [param.reshape(shape) for param, shape in zip(param_list, self.shapes)]
    
    # TODO : make this more general than Gaussian
    def apply_logpdf(self, x, parameters):
        mu_params, sigma_params = self.coerce_parameters(parameters = parameters)
        return self.logpdf(mu_params, sigma_params)(x)
    
    def pdf(self, x, parameters):
        log_vals = self.apply_logpdf(x, parameters)
        return jnp.exp(log_vals)
    
    def fisher_metric(self, parameters, mesh_steps_per_dim = 100, grid_range = (-10, 10)):
        """
        The Fisher metric is built into statistical manifolds.
        """
        # make the mesh
        stepsize = (grid_range[1] - grid_range[0]) / mesh_steps_per_dim
        stepvol = stepsize**self.dim
        mesh_grid = jnp.meshgrid(*[jnp.linspace(*grid_range, mesh_steps_per_dim) for _ in range(self.dim)])
        mesh_grid = jnp.stack(mesh_grid, axis = 0)

        # gradient wrt to the parameters, not the data
        grad_logpdf = grad(self.apply_logpdf, argnums = 1)
        grad_logpdfs = jnp.apply_along_axis(grad_logpdf, 0, mesh_grid, parameters)

        pdfs = jnp.apply_along_axis(self.pdf, 0, mesh_grid, parameters)

        # einsum over most of the dimensions
        fisher_metric = jnp.einsum("i...,j...,...->ij", grad_logpdfs, grad_logpdfs, pdfs) * stepvol

        return fisher_metric

class AlphaConnection(Connection):
    def __init__(self, manifold):
        self.manifold = manifold

def multivariate_gaussian_logpdf(mu_params, sigma_params, eps = 1e-8):
    dim = mu_params.shape[0]
    K = jnp.sqrt(((2 * jnp.pi)**dim) * jnp.linalg.det(sigma_params))
    return lambda x: -jnp.log(K + eps) - 0.5 * ((x - mu_params).T @ jnp.linalg.inv(sigma_params) @ (x - mu_params))

if __name__ == "__main__":
    dim = 3
    key = PRNGKey(0)

    parameters = normal(key, shape = (dim,))

    covariance_params = chisquare(key, 1, shape = (dim, dim))
    covariance_params += covariance_params.T

    variance_params = chisquare(key, 2, shape = (dim,))
    variance_params = jnp.diag(variance_params)
    variance_params += covariance_params * 0.05

    parameters = jnp.concatenate([parameters, variance_params.flatten()])
    manifold = StatisticalManifold(dim,
                                   dim + dim**2,
                                   multivariate_gaussian_logpdf,
                                   [(dim,), (dim, dim)])
    
    test = manifold.fisher_metric(parameters)

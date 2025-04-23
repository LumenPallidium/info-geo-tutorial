from math import prod
import jax.numpy as jnp
from jax.random import PRNGKey, normal, chisquare, binomial
from jax import grad, jacrev
from tqdm import tqdm
from manifolds import *

def generate_mesh(dim, mesh_steps_per_dim = 100, grid_range = (-10, 10)):
    """
    Generate a mesh grid for a given dimension. Note this returns a tensor of shape (dim, mesh_steps_per_dim, ..., mesh_steps_per_dim)
    where mesh_steps_per_dim is repeated dim times.
    """
    stepsize = (grid_range[1] - grid_range[0]) / mesh_steps_per_dim
    stepvol = stepsize**dim
    mesh_grid = jnp.meshgrid(*[jnp.linspace(*grid_range, mesh_steps_per_dim) for _ in range(dim)])
    mesh_grid = jnp.stack(mesh_grid, axis = 0)
    return mesh_grid, stepvol

def hamiltonian_monte_carlo(stat_manifold,
                            eval_function,
                            parameters,
                            n_points = 20,
                            n_steps = 1000,
                            burn_in = 100,
                            step_size = 0.0001,
                            inverse_mass = None,
                            prng_key = None,
                            energy_tolerance = 0.01):
    """
    Performs Hamiltonian MCMC with no-uturn sampling.

    Parameters
    ----------
    stat_manifold : StatisticalManifold
        A statistical manifold object to perform the sampling on.
    eval_function: function
        The function that will have its expectation E(f) computed. For example,
        to compute mean, we would use f(x) = x.
    parameters : jnp.array
        The parameters of the statistical manifold.
    n_points : int
        Number of initial points that will simultaneously be used for sampling.
    n_steps : int
        Number of steps to run HMC for.
    burn_in : int
        Number of steps to discard as transients.
    step_size : float
        Step size for updates.
    inverse_mass : jnp.array
        The inverse mass matrix for computing "kinetic energy". Defaults to identity.
    energy_tolerance : float
        How close start and end energy can be to be considered the same.
    """
    dim = stat_manifold.dim
    if inverse_mass is None:
        inverse_mass = jnp.eye(stat_manifold.dim)
        mass = jnp.eye(stat_manifold.dim)
    else:
        mass = jnp.linalg.inv(inverse_mass)
    if prng_key is None:
        prng_key = PRNGKey(0)
    points = normal(prng_key, shape = (n_points, dim))
    # TODO : add uturn sampler
    leapfrog_steps = 20

    # energy function is -logpdf, grad is force
    grad_logpdf = grad(stat_manifold.apply_logpdf, argnums = 0)
    force_f = lambda x : jnp.apply_along_axis(grad_logpdf, 1, x, parameters)
    batch_logpdf = lambda x : jnp.apply_along_axis(stat_manifold.apply_logpdf,
                                                   1, x, parameters)
    function_value_sum = None
    n_evals = 0
    points_lis = []

    for step in tqdm(range(n_steps)):
        points_lis.append(points.copy())
        momentum = jnp.einsum("ij,nj->ni",
                        mass,
                        normal(prng_key, shape = (n_points, dim)))
        start_energies = batch_logpdf(points)
        start_energies += jnp.einsum("mi,ij,mj->m",
                                     momentum,
                                     inverse_mass,
                                     momentum)

        new_points = points.copy()
        half_momentum = momentum.copy() + 0.5 * step_size * force_f(new_points)
        for l_step in range(leapfrog_steps - 1):
            new_points += step_size * jnp.einsum("ij,mj->mi",
                                                 inverse_mass,
                                                 half_momentum)
            half_momentum += step_size * force_f(new_points)
        
        # need one more full and half step
        new_points += step_size * jnp.einsum("ij,mj->mi",
                                             inverse_mass,
                                             half_momentum)
        new_momentum = half_momentum + 0.5 * step_size * force_f(new_points)

        new_energies = batch_logpdf(new_points)
        new_energies += jnp.einsum("mi,ij,mj->m",
                                   new_momentum,
                                   inverse_mass,
                                   new_momentum)
        #TODO it always diverges.. also should non-diverging points get sent through?
        not_diverged = jnp.isclose(new_energies, start_energies, atol = energy_tolerance)
        if jnp.any(not_diverged):
            accept_probs = jnp.minimum(1, jnp.exp(-new_energies + start_energies))
            accepts = binomial(prng_key, 1, accept_probs).astype(jnp.bool_) & not_diverged
            new_points = new_points.at[~accepts].set(points[~accepts])
            points = new_points.copy()
            if step > burn_in:
                function_value = jnp.apply_along_axis(eval_function,
                                                      1, points, parameters)
                if function_value_sum is not None:
                    function_value_sum += function_value.mean(axis = 0)
                else:
                    function_value_sum = function_value.mean(axis = 0)
                n_evals += 1

    return function_value_sum / n_evals, points_lis

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
    
    def _fisher_metric_grid(self, grad_logpdf, parameters,
                            mesh_steps_per_dim = 100, grid_range = (-10, 10)):
        mesh_grid, stepvol = generate_mesh(self.dim, mesh_steps_per_dim, grid_range)
        grad_logpdfs = jnp.apply_along_axis(grad_logpdf, 0, mesh_grid, parameters)

        pdfs = jnp.apply_along_axis(self.pdf, 0, mesh_grid, parameters)
        # einsum over most of the dimensions
        fisher_metric = jnp.einsum("i...,j...,...->ij", grad_logpdfs, grad_logpdfs, pdfs) * stepvol
        return fisher_metric
    
    def _fisher_metric_hmc(self, grad_logpdf, parameters):
        def fisher_metric_fn(x, parameters):
            grad_matrix = grad_logpdf(x, parameters)
            pointwise_fisher = jnp.einsum("...i,...j->...ij",
                                          grad_matrix, grad_matrix)
            return pointwise_fisher
        fisher_metric = hamiltonian_monte_carlo(self,
                                                fisher_metric_fn,
                                                parameters)
        return fisher_metric
    
    # TODO: this should be extended to (maybe) adaptive quadrature and Hamiltonian MCMC
    #https://en.wikipedia.org/wiki/Adaptive_quadrature
    def fisher_metric(self, parameters, method = "hmc"):
        """
        The Fisher metric is built into statistical manifolds.
        """

        # gradient wrt to the parameters, not the data
        grad_logpdf = grad(self.apply_logpdf, argnums = 1)

        if method == "hmc":
            fisher_metric = self._fisher_metric_hmc(grad_logpdf,
                                                    parameters)
        elif method == "grid":
            fisher_metric = self._fisher_metric_grid(grad_logpdf,
                                                     parameters)
        else:
            raise ValueError(f"Invalid method: {method}")


        return fisher_metric

class AlphaConnection(Connection):
    def __init__(self, stat_manifold, alpha = 1):
        self.alpha = min(max(alpha, 0), 1)

        connection_function = self._generate_connection_function(stat_manifold)
        super().__init__(stat_manifold, connection_function = connection_function)

    def _generate_connection_function(self, stat_manifold, mesh_steps_per_dim = 100, grid_range = (-10, 10)):
        grad_logpdf = grad(stat_manifold.apply_logpdf, argnums = 1)
        hess_logpdf = jacrev(grad_logpdf, argnums = 1)
        
        def connection_function(parameters):
            mesh_grid, stepvol = generate_mesh(self.dim, mesh_steps_per_dim, grid_range)
            hess_logpdfs = jnp.apply_along_axis(hess_logpdf, 0, mesh_grid, parameters)
            grad_logpdfs = jnp.apply_along_axis(grad_logpdf, 0, mesh_grid, parameters)

            pdfs = jnp.apply_along_axis(stat_manifold.pdf, 0, mesh_grid, parameters)

            # einsum over most of the dimensions
            coefficient = jnp.einsum("ij...,k...,...->ijk", hess_logpdfs, grad_logpdfs, pdfs) * stepvol
            if self.alpha != 1:
                a_term = jnp.einsum("i...,j...,k...,...->ijk", grad_logpdfs, grad_logpdfs, grad_logpdfs, pdfs) * stepvol
                coefficient += 0.5 * (1 - self.alpha) * a_term

            return coefficient
        
        return connection_function

#TODO : maybe make each pdf a class with its own logpdf and coerce_parameters method
def multivariate_gaussian_logpdf(mu_params, sigma_params, eps = 1e-8):
    dim = mu_params.shape[0]
    K = ((2 * jnp.pi)**(dim) * jnp.linalg.det(sigma_params))
    return lambda x: -0.5 * (jnp.log(K + eps) + jnp.einsum("i,ij,j->",
                                                          x - mu_params,
                                                          jnp.linalg.inv(sigma_params),
                                                          x - mu_params))

if __name__ == "__main__":
    dim = 2
    bias = 5
    key = PRNGKey(0)

    parameters = normal(key, shape = (dim,)) + bias

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
    
    test, points = manifold.fisher_metric(parameters)
    test2 = manifold.fisher_metric(parameters, method = "grid")
    print("Estimated Fisher metric:\n", test, test2)

    import matplotlib.pyplot as plt
    points = jnp.vstack(points)
    points = points.reshape(-1, 20, 2)

    for i in range(20):
        plt.plot(points[:, i, 0], 
                 points[:, i, 1], 
                 c = "black")

    # plot the logpdf as a contour plot in the background
    x = jnp.linspace(-10, 10, 100)
    y = jnp.linspace(-10, 10, 100)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.zeros((100, 100))
    #TODO : better way to do this?
    for i in range(100):
        for j in range(100):
            Z = Z.at[i, j].set(manifold.apply_logpdf(jnp.array([X[i, j], Y[i, j]]), parameters))
    plt.contourf(X, Y, Z, levels = 20, cmap = "jet", alpha = 0.5)
    plt.colorbar()
    plt.title("Hamiltonian Monte Carlo Sampling")


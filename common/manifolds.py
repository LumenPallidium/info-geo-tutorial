import jax.numpy as jnp
from jax import grad, jacrev

class CoordinateSystem:
    def __init__(self, dim):
        self.dim = dim

    def to_cartesian(self, x):
        raise NotImplementedError
    
    def from_cartesian(self, x):
        raise NotImplementedError
    
    def jacobian(self, x):
        raise NotImplementedError
    
class Cartesian(CoordinateSystem):
    def __init__(self, dim):
        super().__init__(dim)
    
    def to_cartesian(self, x):
        return x
    
    def from_cartesian(self, x):
        return x
    
    def jacobian(self, x, alt_coords = None):
        if alt_coords is None:
            return jnp.eye(self.dim)
        else:
            grad_fn = jacrev(alt_coords.from_cartesian)
            return grad_fn(x)
    
class NSpherical(CoordinateSystem):
    """
    See here:
    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    """
    def __init__(self, dim):
        super().__init__(dim)
    
    def to_cartesian(self, x):
        if self.dim == 2:
            return x[0] * jnp.array([jnp.cos(x[1]), jnp.sin(x[1])])
        else:
            return x[0] * jnp.array([jnp.cos(x[1])] + [jnp.cos(x[i]) * jnp.prod(jnp.sin(x[2:i])) for i in range(2, self.dim)]
                                    + [jnp.prod(jnp.sin(x[1:self.dim]))])
    
    def from_cartesian(self, x):
        r = jnp.linalg.norm(x)
        if self.dim == 2:
            return jnp.array([r, jnp.arctan2(x[1], x[0])])
        else:
            return jnp.array([r] + [jnp.arctan2(jnp.linalg.norm(x[i+1:]), x[i]) for i in range(1, self.dim)])
    
    def jacobian(self, x, alt_coords = None):
        """
        Get the jacobian of the coordinate system at a point.
        If alt_coords is specified then:
        letting c be cartesian coordinates, x being this coordinate system,
        and y being the other coordinate system, we compute:
        dy / dx = dy / dc @ dc / dx
        """
        if alt_coords is None:
            return jnp.eye(self.dim)
        else:
            x_cart = self.to_cartesian(x)
            grad_fn_self = jacrev(self.to_cartesian)
            grad_fn_other = jacrev(alt_coords.from_cartesian)
            return grad_fn_other(x_cart) @ grad_fn_self(x)
    
class Manifold:
    def __init__(self, dim, embedding_dim = None, coordinate_system = Cartesian):
        self.dim = dim
        if embedding_dim is None:
            embedding_dim = dim
        self.embedding_dim = embedding_dim
        self.coordinate_system = coordinate_system(embedding_dim)

    def contains(self, x):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    def get_tangent_space(self, x):
        raise NotImplementedError
    
class NSphere(Manifold):
    def __init__(self,
                 dim,
                 radius = 1,
                 coordinate_system = Cartesian):
        super().__init__(dim - 1, dim, coordinate_system)
        self.radius = radius
    
    def contains(self, x):
        x = self.coordinate_system.to_cartesian(x)
        return jnp.allclose(jnp.linalg.norm(x).sum(), self.radius ** 2)
    
    def sample(self):
        x = jnp.random.randn(self.dim)
        x = self.radius * x / jnp.linalg.norm(x)
        return self.coordinate_system.from_cartesian(x)
    
    def get_tangent_space(self, x, alt_coords = None):
        assert self.contains(x), "Point not on the manifold"
        if alt_coords is None:
            return jnp.eye(self.dim)
        else:
            return self.coordinate_system.jacobian(x, alt_coords)
        
class VectorField:
    def __init__(self, manifold, dim):
        self.manifold = manifold
        self.dim = dim
        self.components = jnp.zeros((dim,))
        self.cartesian = Cartesian(dim)

    def __call__(self, f, p):
        x = self.manifold.coordinate_system.to_cartesian(p)
        grad_f = grad(f)
        # assume f is defined in cartesian coordinates
        df_dx = grad_f(x)
        dx_dp = self.cartesian.jacobian(x,
                                        alt_coords = self.manifold.coordinate_system)
        df_dp = dx_dp @ df_dx
        return df_dp

class TensorField:
    def __init__(self,
                 manifold,
                 dim,
                 components = None,
                 covariant_degree = 0,
                 contravariant_degree = 0):
        self.manifold = manifold
        self.dim = dim
        self.covariant_degree = covariant_degree
        self.contravariant_degree = contravariant_degree
        if components is None:
            components = jnp.zeros((dim,) * covariant_degree + (dim,) * contravariant_degree)
        assert components.shape == (dim,) * covariant_degree + (dim,) * contravariant_degree, "Component shapes don't match tensor degree"
        self.components = components

    def __call__(self, f, x):
        raise NotImplementedError
    
class Metric:
    def __init__(self, manifold : Manifold):
        self.manifold = manifold
        self.dim = manifold.dim
        self.cartesian = Cartesian(self.dim)

    def __call__(self,
                 v1 : VectorField,
                 v2 : VectorField,
                 p):
        g_ij = self.metric_at_point(p)
        metric_value = jnp.einsum("i,ij,j",
                                  v1.components,
                                  g_ij,
                                  v2.components)
        return metric_value
    
    def metric_at_point(self, p):
        p_cart = self.manifold.coordinate_system.to_cartesian(p)
        # jacobian of the coordinate system at p
        jacobian = self.cartesian.jacobian(p_cart,
                                           alt_coords=self.manifold.coordinate_system)
        # using that J^T J = g_ij (derived from transformation law and flatness of Cartesian metric)
        g_ij = jnp.einsum("ik,jk",
                          jacobian,
                          jacobian)
        return g_ij


if __name__ == "__main__":
    # these are cartesian coordinates in R^2
    point_1 = jnp.array([jnp.sqrt(2), jnp.sqrt(2)], dtype=jnp.float32)
    point_2 = -jnp.array([jnp.sqrt(2), jnp.sqrt(2)], dtype=jnp.float32)
    # note these are same points in spherical coordinates
    point_3 = jnp.array([2, jnp.pi / 4], dtype=jnp.float32)
    point_4 = jnp.array([2, 5 * jnp.pi / 4], dtype=jnp.float32) 

    # call these x
    cartesian_coords = Cartesian(2)
    # call these x'
    sphere_coords = NSpherical(2)

    # dx / dx' at point_1
    jacobian_1 = cartesian_coords.jacobian(point_1,
                                           alt_coords=sphere_coords)
    # dx / dx' at point_2
    jacobian_2 = cartesian_coords.jacobian(point_2,
                                           alt_coords=sphere_coords)
    # dx' / dx at point_3 = point_1
    jacobian_3 = sphere_coords.jacobian(point_3,
                                        alt_coords=cartesian_coords)
    # dx' / dx at point_4 = point_2
    jacobian_4 = sphere_coords.jacobian(point_4,
                                        alt_coords=cartesian_coords)

    print(jnp.isclose(jacobian_1 @ jacobian_3, jnp.eye(2)))
    print(jnp.isclose(jacobian_2 @ jacobian_4, jnp.eye(2)))


    #TODO : should result in diag(1, r^2, r^2 sin^2(theta))
    test_manifold = Manifold(3,
                             coordinate_system = NSpherical)
    test_metric = Metric(test_manifold)
    test_point = jnp.array([2, jnp.pi / 4, jnp.pi / 4], dtype=jnp.float32)
    g_ij = test_metric.metric_at_point(test_point)
    print(g_ij)

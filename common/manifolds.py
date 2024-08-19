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
            # d (alt) / d (cart)
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
            return x[0] * jnp.array([jnp.cos(x[1])] + [jnp.cos(x[i]) * jnp.prod(jnp.sin(x[1:i])) for i in range(2, self.dim)]
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
            # d (cart) / d (self)
            grad_fn_self = jacrev(self.to_cartesian)
            # d (alt) / d (cart)
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
    def __init__(self,
                 manifold,
                 dim,
                 components = None):
        self.manifold = manifold
        self.dim = dim
        if components is None:
            components = jnp.ones((dim,))
        # constant function
        if isinstance(components, jnp.ndarray):
            components = lambda _: components
        self.components = components
        self.cartesian = Cartesian(dim)

    def __call__(self, f, p):
        """
        Get the value of the vector field on a function f at point p.

        Parameters
        ----------
        f : callable
            Function to evaluate the vector field on.
        p : jnp.ndarray
            Point to evaluate the vector field at.
        """
        x = self.manifold.coordinate_system.to_cartesian(p)
        components_p = self.components(p)

        grad_f = grad(f)
        # assume f is defined in cartesian coordinates
        df_dx = grad_f(x) * components_p
        dx_dp = self.cartesian.jacobian(x,
                                        alt_coords = self.manifold.coordinate_system)
        df_dp = dx_dp @ df_dx
        return df_dp
    
    def __add__(self, other):
        assert self.manifold == other.manifold, "Manifolds don't match"
        return VectorField(self.manifold,
                           self.dim,
                           lambda x: self.components(x) + other.components(x))
    
    def __mul__(self, scalar):
        return VectorField(self.manifold,
                           self.dim,
                           lambda x: scalar * self.components(x))
    
    def covariant_derivative(self,
                             other,
                             metric):
        def new_components(p):
            christoffels = metric.christoffel_symbols(p)

            #TODO : will need to get contorsion tensor in case of non-symmetric connection e.g.:
            # K = self.manifold.contorsion_tensor(p)
            # christoffels = christoffels + K

            dY_dx = jacrev(other.components)(p)

            components_X = self.components(p)
            components_Y = other.components(p)

            # standard equation for covariant derivative
            new_field = jnp.einsum('j,ji->i',
                                   components_X,
                                   dY_dx)
            new_field += jnp.einsum('ijk,j,k->i',
                                    christoffels,
                                    components_Y,
                                    components_X)
            return new_field
        
        return VectorField(self.manifold,
                           self.dim,
                           new_components)


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
                                  v1.components(p),
                                  g_ij,
                                  v2.components(p))
        return metric_value
    
    def metric_at_point(self, p, covariant = True):
        jacobian = self.manifold.coordinate_system.jacobian(p,
                                                            alt_coords=self.cartesian)
        # using that J^T J = g_ij (derived from transformation law and flatness of Cartesian metric)
        g_ij = jacobian.T @ jacobian
        if not covariant:
            g_ij = jnp.linalg.inv(g_ij)
        return g_ij
    
    def christoffel_symbols(self, p):
        """
        The Christoffel symbols are the connection coefficients of the Levi-Civita connection.

        In standard notation, Γ^m_{ij} corresponds to christoffels[m, i, j].
        """
        g_ij = self.metric_at_point(p)
        g_ij_inv = jnp.linalg.inv(g_ij)
        # note last index is the derivative index
        dg_ij = jacrev(self.metric_at_point)(p)
        
        # standard formula for Christoffel symbols from metric
        #TODO : can clean swapaxes
        christoffels = dg_ij.swapaxes(1, 2) + dg_ij.swapaxes(0, 2) - dg_ij.swapaxes(0, 1)
        christoffels = 0.5 * jnp.einsum('ml,ijl->mij',
                                        g_ij_inv,
                                        christoffels)

        return christoffels
    
class Connection:
    def __init__(self, manifold : Manifold, metric : Metric = None):
        self.manifold = manifold
        #TODO : add support for non-metric connections
        if metric is None:
            raise ValueError("Need a metric to define a connection")
        self.metric = metric

    #TODO this doesn't seem to be working
    def curvature(self, p):
        """
        Classic formula for the Riemann curvature tensor.

        In standard notation, R^l_{ijk} corresponds to curvature[l, i, j, k].
        """
        coefs = self.connection_coefficients(p)
        # index order with Γ^l_{ij} is (l, i, j, {derivative index})
        d_coefs = jacrev(self.connection_coefficients)(p)
        # all terms of the curvature tensor
        curvature = d_coefs.transpose(0, 1, 3, 2) - d_coefs
        curvature += jnp.einsum('hik,lhj->lijk',
                                coefs,
                                coefs)
        curvature -= jnp.einsum('hij,lhk->lijk',
                                coefs,
                                coefs)
        return curvature
    
    def torsion(self, p):
        # metric connection is always symmetric
        if self.metric is not None:
            return jnp.zeros((self.manifold.dim,) * 3)
        else:
            coefs = self.connection_coefficients(p)
            torsion = coefs - coefs.swapaxes(1, 2)
            return torsion
        
    def connection_coefficients(self, p):
        """
        Get the connection coefficients of the Levi-Civita connection.

        In standard notation, Γ^m_{ij} corresponds to connection[m, i, j].
        """
        if self.metric is not None:
            return self.metric.christoffel_symbols(p)
        else:
            raise NotImplementedError
        
    def connection_matrix(self, p1, p2, max_dist = 0.01):
        """
        Get the affine connection matrix between
        tangent spaces at two points using the connection.

        Parameters
        ----------
        p1 : jnp.ndarray
            Starting point.
        p2 : jnp.ndarray
            Ending point.
        max_dist : float
            Maximum distance to travel in one step.
        """
        delta = p2 - p1
        dist = jnp.linalg.norm(delta)
        if dist < max_dist:
            coefs = self.connection_coefficients(p1)
            connection_matrix = jnp.eye(self.manifold.dim)
            connection_matrix -= jnp.einsum('i,kij->kj',
                                            delta,
                                            coefs)
            return connection_matrix
        

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

    # should be identity matrices
    print(jnp.isclose(jacobian_1 @ jacobian_3, jnp.eye(2), atol=1e-5))
    print(jnp.isclose(jacobian_2 @ jacobian_4, jnp.eye(2), atol=1e-5))

    test_manifold = Manifold(3,
                             coordinate_system = NSpherical)
    test_metric = Metric(test_manifold)
    test_point = jnp.array([2, jnp.pi / 4, jnp.pi / 4], dtype=jnp.float32)
    g_ij = test_metric.metric_at_point(test_point).round(4)
    print(g_ij)
    christoffels = test_metric.christoffel_symbols(test_point).round(4)
    print(christoffels)
    test_connection = Connection(test_manifold, test_metric)
    curvature = test_connection.curvature(test_point).round(4)
    print(curvature)
    ricci = jnp.einsum('lilj->ij',
                       curvature).round(4)
    print(ricci)

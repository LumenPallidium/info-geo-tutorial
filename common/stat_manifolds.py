import jax.numpy as jnp
from manifolds import *

class AlphaConnection(Connection):
    def __init__(self, manifold):
        self.manifold = manifold

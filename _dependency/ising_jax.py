import jax
import jax.numpy as jnp
from jax import random

class IsingJAX:
    """
    JAX-native Ising model for use in Numpyro/JAX pipelines.
    """
    def __init__(self, d, mu=None, theta=None, domain=None):
        self.d = d
        self.mu = jnp.zeros(d) if mu is None else mu
        self.theta = jnp.ones((d, d)) if theta is None else theta
        self.domain = jnp.array([-1, 1]) if domain is None else domain

    def set_ferromagnet(self, l, temp, periodic=True, anti=False, isotropic=True):
        """
        Set up the Ising model as a (possibly anisotropic) ferromagnet.
        l must be a tuple or list of two Python ints (static shape, not a JAX array).
        """
        assert isinstance(l, (tuple, list)) and len(l) == 2, "l must be a tuple or list of two ints (static shape)"
        dim = l
        d = self.d
        assert dim[0] * dim[1] == d, "Product of dimensions must equal d"
        # Generate lattice adjacency matrix (no igraph, do manually)
        l_h, l_v = dim
        A = jnp.zeros((d, d))
        for i in range(d):
            row_i, col_i = divmod(i, l_v)
            # Right neighbor
            if col_i < l_v - 1:
                j = i + 1
                A = A.at[i, j].set(1)
                A = A.at[j, i].set(1)
            elif periodic:
                j = i - (l_v - 1)
                A = A.at[i, j].set(1)
                A = A.at[j, i].set(1)
            # Down neighbor
            if row_i < l_h - 1:
                j = i + l_v
                A = A.at[i, j].set(1)
                A = A.at[j, i].set(1)
            elif periodic:
                j = i - (l_h - 1) * l_v
                A = A.at[i, j].set(1)
                A = A.at[j, i].set(1)
        mu = jnp.zeros(d)
        if isotropic:
            theta = jnp.ones((d, d)) / temp
            theta = theta * jnp.triu(A)
            theta = theta + theta.T
        else:
            theta = jnp.zeros((d, d))
            for i in range(d):
                row_i, col_i = divmod(i, l_v)
                # Right neighbor
                if col_i < l_v - 1:
                    j = i + 1
                    val = 1 / temp[0]
                    theta = theta.at[i, j].set(val)
                    theta = theta.at[j, i].set(val)
                elif periodic:
                    j = i - (l_v - 1)
                    val = 1 / temp[0]
                    theta = theta.at[i, j].set(val)
                    theta = theta.at[j, i].set(val)
                # Down neighbor
                if row_i < l_h - 1:
                    j = i + l_v
                    val = 1 / temp[col_i + 1]
                    theta = theta.at[i, j].set(val)
                    theta = theta.at[j, i].set(val)
                elif periodic:
                    j = i - (l_h - 1) * l_v
                    val = 1 / temp[col_i + 1]
                    theta = theta.at[i, j].set(val)
                    theta = theta.at[j, i].set(val)
        if anti:
            theta = -theta
        theta = (theta + theta.T) / 2
        theta = jnp.triu(theta) + jnp.triu(theta, 1).T
        self.mu = mu
        self.theta = theta
        return

    def sample(self, key, num_iters, num_samples=1, burnin_prop=0.5):
        """
        Run Metropolis MCMC for the Ising model using JAX, with burn-in.
        Instead of returning the final spin configuration, return the acceptance probabilities (probs)
        from the last MCMC step for each chain. Shape: (num_samples, d)
        """
        d = self.d
        n = num_samples
        burnin = int(num_iters * burnin_prop)
        total_steps = num_iters + burnin
        key, subkey = random.split(key)
        x = 2 * random.bernoulli(subkey, 0.5, shape=(n, d)) - 1

        def mcmc_step(x, key):
            key1, key2 = random.split(key)
            inds = random.randint(key1, (n,), 0, d)
            b = self.mu[inds] + jnp.einsum("ij,ij->i", x, self.theta[inds, :])
            probs = jnp.exp(-2 * x[jnp.arange(n), inds] * b)
            probs = jnp.minimum(probs, 1.0)
            accept = random.uniform(key2, (n,)) < probs
            x_new = x.at[jnp.arange(n), inds].set(x[jnp.arange(n), inds] * jnp.where(accept, -1, 1))
            return x_new, probs

        keys = random.split(key, total_steps)
        x_final, probs_hist = jax.lax.scan(mcmc_step, x, keys)
        # Only keep the post-burnin steps
        post_burnin_probs = probs_hist[burnin:]
        # Return the last acceptance probabilities for each chain (shape: [num_samples, d])
        # Since probs is (n,) per step, we can tile or broadcast to (n, d) for a surface
        last_probs = post_burnin_probs[-1]
        # Broadcast to (n, d)
        prob_surface = jnp.tile(last_probs[:, None], (1, d))
        return prob_surface

    def check_valid_input(self, x):
        x = jnp.atleast_2d(x)
        assert x.shape[1] == self.d
        assert jnp.all(jnp.isin(x, self.domain))
        return True

    def neg(self, x, i):
        x = jnp.atleast_2d(x)
        assert i < x.shape[1]
        res = x.at[:, i].set(-x[:, i])
        return res

    def score(self, x):
        x = jnp.atleast_2d(x)
        n, d = x.shape
        b = jnp.tile(self.mu, (n, 1)) + x @ self.theta
        res = 1 - jnp.exp(-2 * x * b)
        return res 
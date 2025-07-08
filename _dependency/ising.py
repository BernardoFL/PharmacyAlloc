from __future__ import division
from .util import *
#==========================================================================
# Import: Libraries
#==========================================================================
class Ising(object):
    """
    Ising model.
    """
    def __init__(self, d, mu=None, theta=None):
        """
        Args:
            d: int, dimension.
            mu: array(d), node potentials.
            theta: array((d, d)), pair-wise potentials.
        """
        if mu is None:
            mu = np.zeros(d)

        if theta is None:
            theta = np.ones((d, d))

        assert len(mu) == d
        assert theta.shape == (d, d)
        assert is_symmetric(theta)

        self.d = d
        self.mu = deepcopy(mu)
        self.theta = deepcopy(theta)
        self.domain = [-1, 1]

        return

    def __repr__(self):
        return "Ising(d = %d, mu = %s, theta = %s, domain = %s)" % \
            (self.d, self.mu, self.theta, self.domain)

    def __str__(self):
        res = "------------------------------\n"
        res += "Ising(d, mu, theta)\n"
        res += "d = %d\n" % self.d
        res += "mu = %s\n" % np.round(self.mu, 3)
        res += "theta = %s\n" % np.round(self.theta, 3)
        res += "domain = %s\n" % self.domain
        res += "------------------------------\n"

        return res

    def check_valid_input(self, x):
        """
        Check whether all elements of the input is in the discrete domain of
            the model.

        Args:
            x: list/array of arbitrary dimensions.
        """
        x = np.atleast_2d(x)

        assert x.shape[1] == self.d  # Check dimension
        assert np.all(np.isin(x, self.domain))  # Check values

        return True

    def set_ferromagnet(self, l, temp, periodic=True, anti=False, isotropic=True):
        """
        Configure the parameters of the model to an (anti-)ferromagnet:
            mu = 0 for all i
            For isotropic case:
                theta = 1/temp for all i, j (or -1/T for an anti-ferromagnet)
            For anisotropic case:
                theta_h = 1/temp[0] for horizontal connections
                theta_v = 1/temp[1:] for vertical connections (one per column)

        Args:
            l: int, length of (square-)lattice, d = l^2.
            temp: float or array, temperature parameter(s)
                - float: single temperature for isotropic case
                - array: [theta_h, theta_v1, ..., theta_vC] for anisotropic case
            periodic: boolean, whether the lattice is periodic.
            anti: boolean, ferromagnet or anti-ferromagnet.
            isotropic: boolean, whether to use same parameter for all directions.
        """
        if isinstance(l, int):
            dim = [l, l]
        else:
            assert isinstance(l, list)
            dim = l

        d = self.d
        assert np.prod(dim) == d
        # Generate lattice graph
        g = ig.Graph.Lattice(dim=dim, circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()

        mu = np.zeros(d)
        
        if isotropic:
            # Original isotropic case
            theta = np.ones((d, d)) / temp
            theta = theta * np.triu(A)
            theta = theta + theta.T  # Symmetrize
        else:
            # Anisotropic case with different parameters for horizontal and vertical
            theta = np.zeros((d, d))
            l_h, l_v = dim
            
            # Convert 1D indices to 2D coordinates
            def to_2d(idx):
                return idx // l_v, idx % l_v
                
            # Set horizontal connections (using temp[0])
            for i in range(d):
                row_i, col_i = to_2d(i)
                # Right neighbor
                if col_i < l_v - 1:
                    j = i + 1
                    val = 1/temp[0]
                    theta[i, j] = val
                    theta[j, i] = val  # Ensure symmetry
                # Periodic boundary
                elif periodic:
                    j = i - (l_v - 1)
                    val = 1/temp[0]
                    theta[i, j] = val
                    theta[j, i] = val  # Ensure symmetry
                    
            # Set vertical connections (using temp[1:])
            for i in range(d):
                row_i, col_i = to_2d(i)
                # Down neighbor
                if row_i < l_h - 1:
                    j = i + l_v
                    val = 1/temp[col_i + 1]  # Use column-specific parameter
                    theta[i, j] = val
                    theta[j, i] = val  # Ensure symmetry
                # Periodic boundary
                elif periodic:
                    j = i - (l_h - 1) * l_v
                    val = 1/temp[col_i + 1]
                    theta[i, j] = val
                    theta[j, i] = val  # Ensure symmetry

        if anti:
            theta = -theta

        # Final symmetry check and fix
        theta = (theta + theta.T) / 2
        
        # Debug: Print max difference between theta and its transpose
        max_diff = np.max(np.abs(theta - theta.T))
        print(f"Maximum difference between theta and its transpose: {max_diff}")
        
        # Ensure exact symmetry by copying upper triangle to lower triangle
        theta = np.triu(theta) + np.triu(theta, 1).T

        self.mu = deepcopy(mu)
        self.theta = deepcopy(theta)

        return

    def neg(self, x, i):
        """
        Negate the i-th coordinate of x.
        """
        self.check_valid_input(x)
        x = np.atleast_2d(x)
        assert i < x.shape[1]

        res = deepcopy(x)
        res[:, i] = -res[:, i]  # Flips between +1 and -1

        self.check_valid_input(res)

        return res

    def sample(self, num_iters, num_samples=1, burnin_prop=.5):
        """
        Draw samples from an Ising model via the Metropolis algorithm.
        Works for both isotropic and anisotropic cases.

        Returns:
            samples: list of num_samples samples.
            x_hist: all samples after burn-in period.
        """
        # Vectorized implementation: for drawing independent samples only
        d = self.d
        n = num_samples
        num_iters = int(num_iters)
        num_accept = np.zeros(n)

        x = 2 * rand.binomial(n=1, p=.5, size=(n, d)) - 1
        self.check_valid_input(x)

        for t in range(num_iters):
            inds = rand.choice(d, size=n)

            # Calculate local field for each sample
            b = self.mu[inds] + np.einsum("ij,ij->i", x, self.theta[inds, :])  # (n,)
            
            # Metropolis acceptance probability
            probs = np.exp(-2 * x[range(n), inds] * b)
            probs[probs > 1.] = 1.  # Metropolis
            assert_shape(probs, (n,))

            # Accept/reject moves
            accepted = 1 * (rand.uniform(size=n) < probs)
            signs = 1 - 2 * accepted  # (n,); maps 1 to -1 and 0 to 1
            x[range(n), inds] *= signs  # Flip corresponding bits
            num_accept += accepted  # (n,)
            
            #print('{:2.0%} complete.'.format(t/num_iters), end='\r')
        
        print("Metropolis acceptance rate: {0:.4f}\n".format(np.mean(num_accept)/num_iters))

        self.check_valid_input(x)

        return x  # (n, d)

    def score(self, x):
        """
        Computes the (difference) score function.
        Works for both isotropic and anisotropic cases.
        """
        x = np.atleast_2d(x)
        self.check_valid_input(x)
        n, d = x.shape

        # Calculate local field for each sample
        b = np.tile(self.mu, (n, 1)) + x.dot(self.theta)  # (n, d)
        res = 1 - np.exp(-2 * x * b)

        return res

    def plot_grid(self, x, l, figsize=(8, 6)):
        """
        Creates a plot of x on an l-by-l lattice.
        """
        self.check_valid_input(x)
        assert self.d == l**2  # Square grid

        fig = plt.figure(figsize=figsize)
        ax = sns.heatmap(np.reshape(x, (l, l)), square=True,
                         linewidths=1,  # Grid lines
                         xticklabels=False, yticklabels=False,
                         cmap=sns.cubehelix_palette(), cbar=False)

        return fig
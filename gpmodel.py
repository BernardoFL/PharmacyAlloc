import torch
import gpytorch
import pyro
import pyro.distributions as dist
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
from torch.distributions import Normal

class GPTimeMargModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, condition_list, ninducing):
        q_u = CholeskyVariationalDistribution(ninducing, batch_shape=self.batch_shape)
        variational_strategy = VariationalStrategy(
            self, self.inducing_inputs, q_u,  learn_inducing_locations=True
        )
        super(GPTimeMargModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        composite_condition_kernel = CompositeConditionKernel(condition_list)

        self.covar_module = gpytorch.kernels.ScaleKernel(FullCompositeKernel(composite_condition_kernel))
        
        # Standard normal CDF for probit link function
        self.standard_normal = Normal(0, 1)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Get GP latent function values
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        # Apply probit link function by passing through standard normal CDF
        prob = self.standard_normal.cdf(latent_pred.mean)
        # Return binary prediction
        return torch.bernoulli(prob)


#############################################
# Custom Composite Condition Kernel
#############################################

class FullCompositeKernel(gpytorch.kernels.Kernel):
    """
    Full composite kernel for a GP model over a grid where the first element of the input tensor
    is a condition index and the second element is a continuous feature.
    
    Input: x is a tensor of shape (2, 1)
      - x[0, 0] is the condition index (expected to be numeric; will be converted to float).
      - x[1, 0] is the continuous feature.
    
    The kernel computes:
         K_full(x, x') = K_cond(x[0], x'[0]) * K_rbf(x[1], x'[1])
    where:
         - K_cond is a CompositeConditionKernel (assumed to be defined elsewhere) that compares two conditions,
         - K_rbf is an RBF kernel for the continuous feature.
    """
    def __init__(self, composite_condition_kernel, **kwargs):
        """
        Args:
            composite_condition_kernel: Instance of CompositeConditionKernel.
            rbf_kernel: Instance of gpytorch.kernels.RBFKernel.
        """
        super().__init__(**kwargs)
        self.composite_condition_kernel = composite_condition_kernel

    def forward(self, x1, x2, diag=False, **params):
        # Assume x1 and x2 are each of shape (2, 1)
        # Extract the condition index and continuous feature.
        # Use unsqueeze to convert scalar tensors to (1, 1) tensors.
        cond1 = x1[0, 0].unsqueeze(0).unsqueeze(1).float()  # shape: (1, 1)
        cont1 = x1[1, 0].unsqueeze(0).unsqueeze(1).float()  # shape: (1, 1)
        cond2 = x2[0, 0].unsqueeze(0).unsqueeze(1).float()  # shape: (1, 1)
        cont2 = x2[1, 0].unsqueeze(0).unsqueeze(1).float()  # shape: (1, 1)

        # Compute the condition kernel.
        K_cond = self.composite_condition_kernel(cond1, cond2, diag=diag, **params)

        
        # Return the elementwise product.
        return K_cond 

# Example usage:
# Assume you have already created an instance of CompositeConditionKernel, say:
# composite_condition_kernel = CompositeConditionKernel(condition_list)
# and an RBF kernel:
# rbf_kernel = gpytorch.kernels.RBFKernel()
#
# Then you can create the full kernel as:
# full_kernel = FullCompositeKernel(composite_condition_kernel, rbf_kernel)
#
# Now, if you have two inputs, each with shape (2,1):
# x1 = torch.tensor([[1.0], [3.5]])
# x2 = torch.tensor([[2.0], [4.0]])
#
# Then calling:
# K_val = full_kernel(x1, x2)
# will compute:
#   K_val = K_cond(1, 2) * K_rbf(3.5, 4.0)

class CompositeConditionKernel(gpytorch.kernels.Kernel):
    """
    Kernel that compares two conditions based on the shared ATC levels of their associated drugs.
    
    This kernel expects that x1 and x2 are tensors each containing a single condition index (a scalar).
    It looks up the corresponding Condition objects in its internal condition_list and computes a similarity
    score using the condition_similarity function.
    
    Args:
        condition_list (List[Condition]): A list of Condition objects.
    """
    def __init__(self, condition_list, **kwargs):
        super().__init__(**kwargs)
        self.condition_list = condition_list

    def forward(self, x1, x2, diag=False, **params):
        # Ensure that x1 and x2 are scalar indices (i.e. each has exactly one element)
        if x1.numel() != 1 or x2.numel() != 1:
            raise ValueError("CompositeConditionKernel expects each input tensor to contain a single condition index.")
        
        # Get the condition objects from the list.
        cond1 = self.condition_list[int(x1.item())]
        cond2 = self.condition_list[int(x2.item())]
        
        # Compute the similarity score for the two conditions.
        score = condition_similarity(cond1, cond2)
        # Return the score as a 1x1 tensor.
        return torch.tensor([[score]], device=x1.device, dtype=torch.float32)

def condition_similarity(cond1, cond2):
    """
    Compute a similarity score between two Condition objects.
    
    For each pair of drugs (one from cond1.drugs and one from cond2.drugs), determine the maximum level i 
    (0 ≤ i ≤ 4) such that the first i ATC levels match exactly. For each drug pair, add i to the score.
    The final kernel value is the sum over all such drug pairs.
    
    Args:
        cond1, cond2: Condition objects. Each should have an attribute 'drugs' (a list of Drug objects).
                      Each Drug object is assumed to have an attribute 'atcs', where drug.atcs[0] is a list of 4 integers.
    
    Returns:
        A float representing the summed similarity score.
    """
    total_score = 0.0
    for drug1 in cond1.drugs:
        for drug2 in cond2.drugs:
            # Skip if either drug does not have ATC information.
            if not drug1.atcs or not drug2.atcs:
                continue
            atc1 = drug1.atcs[0]
            atc2 = drug2.atcs[0]
            if len(atc1) != 4 or len(atc2) != 4:
                continue
            match = 0
            # Compare level-by-level until a mismatch occurs.
            for level in range(4):
                if atc1[level] == atc2[level]:
                    match += level + 1 # Add 1 to convert 0-based index to 1-based score.
                else:
                    break
            total_score += match
    return total_score
import torch
import torch.nn as nn
from functools import partial
from gpytorch.utils import linear_cg
import torch.autograd.functional as autograd_F
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

from enflows.transforms import NaiveLinear
# Set a random seed for reproducibility
torch.manual_seed(42)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = self.fc2(x)  # Output layer (no activation)
        return x


class LinearFunction(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearFunction, self).__init__()
        # self.fc = nn.Linear(input_size, output_size)
        self.matrix = torch.nn.Parameter(torch.randn(input_size, output_size), requires_grad=True)

    def forward(self, x):
        orth, _ =torch.linalg.qr(self.matrix)
        return x @ orth
        # return self.fc(x)

# Initialize the network
input_size = 2
hidden_size = 2
output_size = 2
# decoder = LinearFunction(input_size, output_size)
decoder = NaiveLinear(input_size)

# Define an input tensor (example input)
x = torch.randn(1, input_size, requires_grad=True)  # Batch size 1, input_size=3
y = decoder(x)[0]
def decoder_forward(x):
    y, logabsdet = decoder(x)  # Only capture the first output, discard the second
    # y = decoder(x)  # Only capture the first output, discard the second
    return y

# jacobian_ = torch.autograd.functional.jacobian(decoder.forward, x).sum(-2)
jacobian = []
for i in range(output_size):  # Iterate over output dimensions (output_size=2)
    grad_output = torch.zeros_like(y)
    grad_output[0, i] = 1  # We want to compute the gradient with respect to the i-th output
    jacobian_i = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output, create_graph=True)[0]
    jacobian.append(jacobian_i)

# Stack the individual gradients into a full Jacobian matrix
jacobian = torch.stack(jacobian, dim=0).squeeze()

# Print the Jacobian
print("Jacobian Matrix:")
print(jacobian)

def hutchinson_estimate(latent, function, num_hutchinson_samples, hutchinson_distribution="normal", max_cg_iterations=None, cg_tolerance=None, training=True):
    sample_shape = (*latent.shape, num_hutchinson_samples)
    if hutchinson_distribution == "normal":
        hutchinson_samples = torch.randn(*sample_shape, device=latent.device)
    elif hutchinson_distribution == "rademacher":
        bernoulli_probs = 0.5 * torch.ones(*sample_shape, device=latent.device)
        hutchinson_samples = torch.bernoulli(bernoulli_probs)
        hutchinson_samples.mul_(2.).subtract_(1.)

    repeated_latent = latent.repeat_interleave(num_hutchinson_samples, dim=0)

    def jvp_forward(x, v, function):
        with dual_level():
            inp = make_dual(x, v)
            out = function(inp)
            y, jvp = unpack_dual(out)

        return y, jvp

    def jac_transpose_jac_vec(latent, vec, create_graph, function):
        if not create_graph:
            latent = latent.detach().requires_grad_(False)
            with torch.no_grad():
                y, jvp = jvp_forward(latent, vec, function)
        else:
            y, jvp = jvp_forward(latent, vec, function)

        flow_forward_flat = lambda x: function(x).flatten(start_dim=1) # possible mistake here
        _, jtjvp = autograd_F.vjp(flow_forward_flat, latent, jvp.flatten(start_dim=1), create_graph=create_graph)

        return jtjvp, y

    def tensor_to_vector(tensor):
        # Turn a tensor of shape (batch_size x latent_dim x num_hutch_samples)
        # into a vector of shape (batch_size*num_hutch_samples x latent_dim)
        # NOTE: Need to transpose first to get desired stacking from reshape
        vector = tensor.transpose(1, 2).reshape(
            x.shape[0] * num_hutchinson_samples, x.shape[1]
        )
        return vector

    def vector_to_tensor(vector):
        # Inverse of `tensor_to_vector` above
        # NOTE: Again need to transpose to correctly unfurl num_hutch_samples as the final dimension
        tensor = vector.reshape(x.shape[0], num_hutchinson_samples, x.shape[1])
        return tensor.transpose(1, 2)

    def jac_transpose_jac_closure(tensor, function):
        # NOTE: The CG method available to us expects a method to multiply against
        #       tensors of shape (batch_size x latent_dim x num_hutch_samples).
        #       Thus we need to wrap reshaping around our JtJv method,
        #       which expects v to be of shape (batch_size*num_hutch_samples x latent_dim).
        vec = tensor_to_vector(tensor)
        jtjvp, _ = jac_transpose_jac_vec(repeated_latent, vec, create_graph=False, function=function)
        return vector_to_tensor(jtjvp)

    jtj_inverse_hutchinson = linear_cg(
        partial(jac_transpose_jac_closure, function=function),
        hutchinson_samples,
        max_iter=max_cg_iterations,
        max_tridiag_iter=max_cg_iterations,
        tolerance=cg_tolerance
    ).detach()

    jtj_hutchinson_vec, reconstruction_repeated = jac_transpose_jac_vec(
        repeated_latent, tensor_to_vector(hutchinson_samples), create_graph=training, function=function
    )
    reconstruction = reconstruction_repeated[::num_hutchinson_samples]
    jtj_hutchinson = vector_to_tensor(jtj_hutchinson_vec)

    # NOTE: jtj_inverse does not just cancel out with jtj because the former has a stop gradient applied.
    approx_log_det_jac = torch.mean(torch.sum(jtj_inverse_hutchinson * jtj_hutchinson, dim=1, keepdim=True), dim=2)

    return approx_log_det_jac, reconstruction

num_hutchinson_samples = 100000
approx_log_det_jac, reconstruction = hutchinson_estimate(x, function=decoder_forward, num_hutchinson_samples=num_hutchinson_samples,
                                                         hutchinson_distribution="normal", max_cg_iterations=None, cg_tolerance=None, training=True)

exact_log_det_jac = 0.5 * torch.logdet(jacobian.mT @ jacobian)

grad_approx_logdet = torch.autograd.grad(approx_log_det_jac, decoder.parameters(), allow_unused=True)
grad_exact_logdet = torch.autograd.grad(exact_log_det_jac, decoder.parameters(), allow_unused=True)

n_samples = [1, 5, 10, 25, 50, 100, 500, 1000, 5000, 10_000, 100_000, 1_000_000]
hutch = [hutchinson_estimate(x, function=decoder_forward, num_hutchinson_samples=i,hutchinson_distribution="normal", max_cg_iterations=None, cg_tolerance=None, training=True)[0]
         for i in n_samples]
hutch_grad = [torch.autograd.grad(hutch[i], decoder.parameters(), allow_unused=True)[1] for i in range(len(n_samples))]
diff = [torch.norm(grad_exact_logdet[1]-hutch_grad[i]).item() for i in range(len(n_samples))]
import matplotlib.pyplot as plt
plt.plot(n_samples, diff)
plt.xscale("log")
plt.show()
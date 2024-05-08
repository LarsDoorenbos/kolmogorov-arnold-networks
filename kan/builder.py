
import logging

import torch
import torch.nn as nn
import ignite.distributed as idist

import numpy as np
import einops

LOGGER = logging.getLogger(__name__)


class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_intervals, spline_order):
        super(KANLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_intervals = num_intervals
        self.spline_order = spline_order
        self.num_control_points = num_intervals + spline_order

        initial_domain_ranges = torch.as_tensor([[-1, 1]]).repeat(in_dim * out_dim, 1).reshape(in_dim, out_dim, -1).to(idist.device())
        initialization_variance = 0.1

        # Parameters for in_dim * out_dim activation functions
        # Initialize control points for all activations
        self.control_points = nn.Parameter(torch.as_tensor(np.random.normal(0, initialization_variance, (in_dim, out_dim, self.num_control_points)), dtype=torch.float32), requires_grad=True)

        # Initialize grid per activation function
        grids = self.make_grid(initial_domain_ranges)
        self.register_buffer("grids", grids)

        # Residual function silu
        self.silu = nn.SiLU()

        # Initialize weights of the scaling factors for both the residual connections and the splines with Xavier initialization. 
        # In the current version of the paper (8th of May) they have only one factor, but it should be two as confirmed by the authors (https://github.com/KindXiaoming/pykan/issues/128#issuecomment-2100761400)
        xavier_weights_res = np.random.normal(0, np.sqrt(2 / out_dim), (in_dim, out_dim))
        self.residual_scaling = nn.Parameter(torch.as_tensor(xavier_weights_res, dtype=torch.float32), requires_grad=True)

        xavier_weights_spline = np.random.normal(0, np.sqrt(2 / out_dim), (in_dim, out_dim))
        self.spline_scaling = nn.Parameter(torch.as_tensor(xavier_weights_spline, dtype=torch.float32), requires_grad=True)

        self.grid_margin = 0.01

    def make_grid(self, domains):
        step_sizes = (domains[:, :, 1] - domains[:, :, 0]) / self.num_intervals
        extended_grid = (domains[:, :, 0] - self.spline_order * step_sizes).unsqueeze(-1) + torch.arange(0, self.num_intervals + 2 * self.spline_order + 1, device=idist.device()).reshape(1, 1, -1) * step_sizes[:, :, None]
        
        return extended_grid.to(idist.device())
        
    def batched_cox_de_boor(self, i, degree, x):
        expanded_x = x.unsqueeze(-1)

        if degree == 0:
            return (self.grids[:, :, i] <= expanded_x) * (expanded_x < self.grids[:, :, i + 1])
        
        left_side = (expanded_x - self.grids[:, :, i]) / (self.grids[:, :, i + degree] - self.grids[:, :, i]) * self.batched_cox_de_boor(i, degree - 1, x)
        right_side = (self.grids[:, :, i + degree + 1] - expanded_x) / (self.grids[:, :, i + degree + 1] - self.grids[:, :, i + 1]) * self.batched_cox_de_boor(i + 1, degree - 1, x)

        return left_side + right_side

    def compute_spline(self, x):
        # TODO: make vectorized version
        result = 0.0
        for i in range(self.num_intervals + self.spline_order):
            result += self.control_points[:, :, i] * self.batched_cox_de_boor(i, self.spline_order, x)

        return result

    def forward(self, x):
        # Compute activations
        spline = self.compute_spline(x)

        # Add residual connection and scaling
        res_x = self.silu(x)
        activations = self.residual_scaling * res_x.unsqueeze(-1) + self.spline_scaling * spline

        # Regularization
        l1_norms = torch.sum(torch.abs(activations), dim=0) / x.shape[0]
        
        l1 = torch.sum(l1_norms)
        entropy = -torch.sum((l1_norms / l1) * torch.log(l1_norms / l1))

        # Sum results into neurons for next layer
        result = activations.reshape(x.shape[0], self.in_dim, self.out_dim)
        result = torch.sum(result, dim=1)

        return result, l1, entropy, activations
    
    def update_grid(self, x, update, refine):
        old_out, _, _, _ = self.forward(x)
        old_acts = self.compute_spline(x)

        if refine:
            self.num_intervals *= 2
            self.num_control_points = self.num_intervals + self.spline_order
        
        if update:
            # Get ranges of activations
            min_acts = torch.min(x, dim=0).values - self.grid_margin
            max_acts = torch.max(x, dim=0).values + self.grid_margin

            # Update grid
            domains = torch.stack([min_acts, max_acts], dim=1).unsqueeze(1).repeat(1, self.out_dim, 1)
            self.grids.data = self.make_grid(domains).to(idist.device())
        else:
            # Refine grid without recomputing bounds
            domains = torch.stack((self.grids[:, :, self.spline_order], self.grids[:, :, -(self.spline_order + 1)]), dim=-1)
            self.grids.data = self.make_grid(domains).to(idist.device())
        
        # Update control points
        # TODO: make vectorized version
        cdbs = []
        for i in range(self.num_intervals + self.spline_order):
            cdbs.append(self.batched_cox_de_boor(i, self.spline_order, x))

        cdbs = torch.stack(cdbs)

        cdbs = einops.rearrange(cdbs, 'cps bs i o -> i o bs cps')
        old_acts = einops.rearrange(old_acts, 'bs i o -> i o bs').unsqueeze(-1)

        new_control_points = torch.linalg.lstsq(cdbs, old_acts).solution.squeeze(-1)

        # Check if any nan or infs
        if torch.isnan(new_control_points).any() or torch.isinf(new_control_points).any():
            LOGGER.warning("NaN or inf detected in control points. Falling back to CPU.")
            new_control_points = torch.linalg.lstsq(cdbs.cpu(), old_acts.cpu(), driver='gelss').solution.squeeze(-1).to(idist.device())

        self.control_points.data = new_control_points.to(idist.device())

        return old_out
    

class KAN(nn.Module):
    def __init__(self, in_dim, out_dim, params):
        super(KAN, self).__init__()

        num_intervals = params["num_intervals"]
        spline_order = params["spline_order"]
        hidden_layer_dims = params["hidden_layer_dims"]

        in_channels = [in_dim] + list(hidden_layer_dims)
        out_channels = list(hidden_layer_dims) + [out_dim]

        self.layers = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels, out_channels):
            self.layers.append(KANLayer(in_ch, out_ch, num_intervals, spline_order))
        
    def forward(self, x):
        l1_reg = 0
        entropy_reg = 0
        for layer in self.layers:
            x, l1, entropy, _ = layer(x)

            l1_reg += l1
            entropy_reg += entropy
        
        return x, l1_reg, entropy_reg
    
    @torch.no_grad()
    def update_grids(self, x, update, refine):
        # Update grids
        for layer in self.layers:
            x = layer.update_grid(x, update, refine)

    @torch.no_grad()
    def get_all_activations(self, x):
        inputs = []
        activations = []
        for layer in self.layers:
            inputs.append(x.cpu().numpy())

            x, _, _, acts = layer(x)
            
            activations.append(acts.cpu().numpy())
        
        return inputs, activations


def build_model(params, input_shape, output_shape):
    model = KAN(input_shape[0], output_shape[0], params)
    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("Trainable params: %d", num_of_parameters)

    return model
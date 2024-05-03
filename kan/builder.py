
import logging

import torch
import torch.nn as nn
import ignite.distributed as idist

import numpy as np

LOGGER = logging.getLogger(__name__)


class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_intervals, spline_order):
        super(KANLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_intervals = num_intervals
        self.spline_order = spline_order
        
        initial_domain = [-1, 1]
        num_control_points = num_intervals + spline_order
        variance = 0.1

        # Parameters for in_dim * out_dim activation functions
        # Initialize control points for all activations
        self.control_points = nn.Parameter(torch.as_tensor(np.random.normal(0, variance, (in_dim, out_dim, num_control_points))), requires_grad=True)

        # Initialize grid per activation function
        grid = torch.linspace(initial_domain[0], initial_domain[1], num_intervals + 1)
        step_size = (initial_domain[1] - initial_domain[0]) / num_intervals

        pre_padding = torch.linspace(initial_domain[0] - spline_order * step_size, initial_domain[0] - step_size, spline_order)
        post_padding = torch.linspace(initial_domain[1] + step_size, initial_domain[1] + spline_order * step_size, spline_order)
        extended_grid = torch.cat((pre_padding, grid, post_padding))

        self.grids = extended_grid.repeat(in_dim * out_dim, 1).reshape(in_dim, out_dim, -1).to(idist.device())

        # Residual function silu
        self.silu = nn.SiLU()

        # Initialize weights for the scaling factor with Xavier initialization
        xavier_weights = np.random.normal(0, np.sqrt(2 / out_dim), (in_dim, out_dim))
        self.scaling_factors = nn.Parameter(torch.as_tensor(xavier_weights), requires_grad=True)
        
    def batched_cox_de_boor(self, i, degree, x):
        expanded_x = x.unsqueeze(-1)

        if degree == 0:
            return (self.grids[:, :, i] <= expanded_x) * (expanded_x < self.grids[:, :, i + 1])
        
        left_side = (expanded_x - self.grids[:, :, i]) / (self.grids[:, :, i + degree] - self.grids[:, :, i]) * self.batched_cox_de_boor(i, degree - 1, x)
        right_side = (self.grids[:, :, i + degree + 1] - expanded_x) / (self.grids[:, :, i + degree + 1] - self.grids[:, :, i + 1]) * self.batched_cox_de_boor(i + 1, degree - 1, x)

        return left_side + right_side
    
    def forward(self, x):
        # Compute activations
        result = 0.0
        for i in range(self.num_intervals + self.spline_order):
            result += self.control_points[:, :, i] * self.batched_cox_de_boor(i, self.spline_order, x)

        # Add residual connection and scaling
        res_x = self.silu(x)
        result = self.scaling_factors * (res_x.unsqueeze(-1) + result)

        # Sum results into neurons for next layer
        result = result.reshape(x.shape[0], self.in_dim, self.out_dim)
        result = torch.sum(result, dim=1)

        # TODO: cast before so computations are done faster?
        return result.to(torch.float32)
    

class KAN(nn.Module):
    def __init__(self, in_dim, out_dim, num_intervals, spline_order, hidden_layer_dims):
        super(KAN, self).__init__()

        in_channels = [in_dim] + list(hidden_layer_dims)
        out_channels = list(hidden_layer_dims) + [out_dim]

        self.layers = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels, out_channels):
            self.layers.append(KANLayer(in_ch, out_ch, num_intervals, spline_order))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x


def build_model(params, input_shape, output_shape):
    model = KAN(input_shape[0], output_shape[0], params["num_intervals"], params["spline_order"], params["hidden_layer_dims"])
    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("Trainable params: %d", num_of_parameters)

    return model
"""Implements a full residual block around a black box layer.

Configurable options include:
normalization position: prenorm or postnorm
normalization type: batchnorm, layernorm etc.
subsampling/pooling
residual options: feedforward, residual, affine scalars, depth-dependent scaling, etc.
"""
import random
from functools import partial
import torch
from torch import nn
from src.models.nn import LinearActivation, Activation, DropoutNd

from src.models.nn import Normalization, StochasticDepth, DropoutNd
from src.models.sequence import SequenceModule
from src.models.sequence.modules.pool import registry as pool_registry
from src.models.nn.residual import registry as residual_registry
import src.utils as utils
import src.utils.registry as registry
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepIFNode
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from torch.autograd import Function


def create_random_mask(size, sparsity=0.9):
    # Generate a mask with 90% of elements being 0
    mask = torch.zeros(size)

    # Number of elements to sample from random normal
    num_non_zero = int(size.numel() * (1 - sparsity))

    # Randomly select positions to set non-zero elements
    non_zero_indices = torch.randperm(size.numel())[:num_non_zero]

    # Set non-zero elements to random values sampled from normal distribution
    mask.view(-1)[non_zero_indices] = 1 #torch.ones(num_non_zero)

    return mask

class SequenceResidualBlock(SequenceModule):
    class Replace(Function):
        @staticmethod
        def forward(ctx, z1, z1_r):
            return z1_r

        @staticmethod
        def backward(ctx, grad):
            return (grad, grad)
    """Flexible residual block design. See model.py for meaning of options."""

    def __init__(
            self,
            d_input,
            i_layer=None, # Only needs to be passed into certain residuals like Decay
            prenorm=True,
            bidirectional=False,
            dropout=0.0,
            tie_dropout=False,
            transposed=False,
            layer=None, # Config for black box module
            residual=None, # Config for residual function
            norm=None, # Config for normalization layer
            pool=None,
            drop_path=0.,
        ):
        super().__init__()

        self.i_layer = i_layer
        self.d_input = d_input
        self.prenorm = prenorm
        self.bidirectional = bidirectional
        self.transposed = transposed
        self.activation = Activation('gelu')
        #print(d_input)
        self.layer = utils.instantiate(registry.layer, layer, d_input)
        self.parametric_prob = True

        if self.parametric_prob:
            self.inp_b = nn.Parameter(torch.zeros(self.layer.d_output), requires_grad=True)
            self.inp_a = nn.Parameter(torch.ones(self.layer.d_output), requires_grad=True)
            self.inp2_b = nn.Parameter(torch.zeros(self.layer.d_output), requires_grad=True)
            self.inp2_a = nn.Parameter(torch.ones(self.layer.d_output), requires_grad=True)
            self.out_b = nn.Parameter(torch.zeros(self.layer.d_output), requires_grad=True)
            self.out_a = nn.Parameter(torch.ones(self.layer.d_output), requires_grad=True)

        if self.bidirectional:
            self.reverse_layer = utils.instantiate(registry.layer, layer, d_input)
            self.bidirectional_linear = nn.Linear(2*self.layer.d_output, self.layer.d_output)
            self.bidirectional_linear1 = nn.Linear(self.layer.d_output, int(self.layer.d_output/2))
            self.bidirectional_linear2 = nn.Linear(self.layer.d_output, int(self.layer.d_output/2))
        else:
            print('Num of neurons: ', self.layer.d_output)
            self.bidirectional_linear1 = nn.Linear(self.layer.d_output, self.layer.d_output)

        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            self.residual = utils.instantiate(residual_registry, residual, i_layer, d_input, self.layer.d_output)
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool
        self.pool = utils.instantiate(pool_registry, pool, self.d_residual, transposed=self.transposed)

        # Dropout
        dropout_cls = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path, mode='row') if drop_path > 0.0 else nn.Identity()
        # sample = torch.randn((128, 2048, 256))
        # self.mask = create_random_mask(sample.size(), sparsity=0.20)

    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def stochSpikeGen(self, y, random_uniform):
        y_spikes = torch.where(y > random_uniform, torch.ones_like(y), torch.zeros_like(
            y))  # + torch.where(y < -1 * random_uniform, -1 * torch.ones_like(y), torch.zeros_like(y))
        # Surrogate
        y = torch.clamp(y, 0, 1)

        y = self.Replace.apply(y, y_spikes)

        return y

    def spikeMixer(self, y, layer):
        y = layer(y)
        y = self.activation(y)

        return y

    def forward(self, x, state=None, **kwargs):
        device = x.device
        #x = x * self.mask[:,:x.shape[1],:].to(device)
        self.PoissonEncoder = encoding.PoissonEncoder()
        size = x.size()

        if self.parametric_prob:
            y =  (self.inp_a * x + self.inp_b)
        else:
            y = x
            # Pre-norm

        if self.norm is not None and self.prenorm: y = self.norm(y)

        random_uniform = torch.rand_like(y)
        random_uniform_2 = torch.rand_like(y)

        y = self.stochSpikeGen(y, random_uniform)

        y_for, new_state = self.layer(y, state=state, **kwargs)

        if self.parametric_prob:
            y_for = (y_for * self.out_a + self.out_b)

        # Adding fault

        if self.bidirectional:
            if self.parametric_prob:
                y = (self.inp2_a * x + self.inp2_b)
            else:
                y = x

            y_spikes = torch.where(y > random_uniform_2, torch.ones_like(y), torch.zeros_like(y)) #+ torch.where(y < -1 * random_uniform, -1 * torch.ones_like(y), torch.zeros_like(y))
            y = torch.clamp(y, 0, 1)
            y = self.Replace.apply(y, y_spikes)

            y_rev, _ = self.reverse_layer(y, state=state, **kwargs)

            y_for_spikes = torch.where(y_for > random_uniform, torch.ones_like(y_for), torch.zeros_like(y_for)) #+ torch.where(y_for < -1 * random_uniform, -1 * torch.ones_like(y_for), torch.zeros_like(y_for))
            y = torch.clamp(y_for, 0, 1)
            y_for = self.Replace.apply(y, y_for_spikes)


            y_rev_spikes = torch.where(y_rev > random_uniform, torch.ones_like(y_rev), torch.zeros_like(y_rev))
            y = torch.clamp(y_rev, 0, 1)
            y_rev = self.Replace.apply(y, y_rev_spikes)

            y = torch.cat([y_for, y_rev], dim=-1)

            y = self.spikeMixer(y, self.bidirectional_linear)
        else:
            y = self.stochSpikeGen(y_for, random_uniform_2)
            y = self.spikeMixer(y, self.bidirectional_linear1)


        # Can use input spikes as well
        if self.residual is not None:
            y = self.residual(x, self.drop_path(self.drop(y)), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm(y)

        if self.pool is not None:
            y, _ = self.pool(y)

        return y, state

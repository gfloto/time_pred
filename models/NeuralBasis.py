import math
import torch
from torch import nn
from einops import rearrange, repeat
from functools import partial

'''
modified version of: https://github.com/lucidrains/siren-pytorch
this version trains ensembles of siren networks in parallel
'''

# helpers
def exists(val):
    return val is not None

# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0.cuda()
    def forward(self, x):
        return torch.sin(self.w0[None, :, None] * x)

def identity(x):
    return x

# siren layer
class Siren(nn.Module):
    def __init__(
        self,
        ensembles,
        dim_in,
        dim_out,
        w0,
        is_first = False,
        use_bias = True,
        activation = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight, bias = [], []
        for i in range(ensembles):
            weight.append(torch.zeros(dim_out, dim_in))
            if use_bias:
                bias.append(torch.zeros(dim_out))

        self.init_(weight, bias, w0 = w0)

        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, w0, c=6):
        dim = self.dim_in
        self.weight, self.bias = [], []

        for i in range(w0.shape[0]):
            w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0[i])
            weight[i].uniform_(-w_std, w_std)

            if len(bias) > 0:
                bias[i].uniform_(-w_std, w_std)

        # convert to nn.Parameter
        self.weight = nn.Parameter(torch.stack(weight, dim=0))
        if len(bias) > 0:
            self.bias = nn.Parameter(torch.stack(bias, dim=0))
        else:
            self.bias = None

    def forward(self, x):
        if self.is_first:
            y = torch.einsum('e o i, b i -> b e o', self.weight, x)
        else:
            y = torch.einsum('e o i, b e i -> b e o', self.weight, x)

        if exists(self.bias):
            y += self.bias
        return self.activation(y)

# siren network
class SirenNet(nn.Module):
    def __init__(
        self,
        ensembles,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0,
        w0_initial = 30.,
        use_bias = True,
        final_activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                ensembles = ensembles,
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
            )

            self.layers.append(layer)

        final_activation = identity if not exists(final_activation) else final_activation

        self.last_layer = Siren(
            ensembles=ensembles,
            dim_in = dim_hidden,
            dim_out = dim_out,
            w0 = w0,
            use_bias = use_bias,
            activation = final_activation
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.last_layer(x)

class ParallelEigenFunction(nn.Module):
    def __init__(
            self,
            ensembles,
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            w0 = 1.,
            w0_initial = 30.,
            use_bias = True,
            final_activation = None,
            dropout = 0.
        ):
        super().__init__()
        self.net = SirenNet(
            ensembles = ensembles,
            dim_in = dim_in,
            dim_hidden = dim_hidden,
            dim_out = dim_out,
            num_layers = num_layers,
            w0 = w0,
            w0_initial = w0_initial,
            use_bias = use_bias,
            final_activation = final_activation,
            dropout = dropout
        )

    # TODO: for variable sized inputs there must be a better way...
    def forward(self, x):
        return self.net(x)

# make domain for eigen-functions
def make_domain(window_size):
    return torch.linspace(0, 1, window_size)[..., None]

# normalize eigen-functions
def normalize(eigenfunc):
    norm = eigenfunc.square().sum(dim=1).sqrt()[:, None]
    return eigenfunc / norm

# compute inner product of eigen-functions to encourage orthonormality
def orthonormality(eigenfunc):
    prod = torch.einsum('t i c, t j c -> i j c', eigenfunc, eigenfunc) 
    prod = rearrange(prod, 'i j c -> c i j')

    prod = torch.triu(prod, diagonal=1) / eigenfunc.shape[0]
    ortho = prod.square().sum() / (prod.shape[-1] * prod.shape[-2])

    return ortho

# add ones to eigen-functions
def add_ones(eigenfunc):
    sh = eigenfunc.shape
    ones = torch.ones(sh[0], 1, sh[2]).cuda()
    return torch.cat([ones, eigenfunc], dim=1)

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.layers = configs.e_layers
        self.dim_hidden = configs.d_model
        self.num_basis_in = configs.num_basis_in
        self.num_basis_out = configs.num_basis_out

        self.is_train = True
        self.eigenfunc_in = None
        self.eigenfunc_out = None

        linear_in = (self.num_basis_in + 1) * self.channels + 4
        linear_out = (self.num_basis_out + 1) * self.channels
        print(f'linear_in: {linear_in}, linear_out: {linear_out}')
        self.linear = nn.Linear(linear_in, linear_out).cuda()

        self.eigen_models = {}
        partial_model = partial(
            ParallelEigenFunction,
            dim_in = 1,
            dim_out = 1,
            dim_hidden = self.dim_hidden,
            num_layers = self.layers,
        )

        # make input model
        w0_in = torch.ones(self.num_basis_in * self.channels).cuda()
        w0_initial_in = torch.linspace(1, self.num_basis_in, self.num_basis_in).cuda()
        
        w0_initial_in = repeat(w0_initial_in, 'e -> e c', c=self.channels)
        w0_initial_in = rearrange(w0_initial_in, 'e c -> (e c)')

        self.eigenmodel_in = partial_model(
            ensembles = self.num_basis_in * self.channels,
            w0 = w0_in,
            w0_initial = w0_initial_in,
        ).cuda()

        # make output model
        w0_out = torch.ones(self.num_basis_out * self.channels).cuda()
        w0_initial_out = torch.linspace(1, self.num_basis_out, self.num_basis_out).cuda()

        w0_initial_out = repeat(w0_initial_out, 'e -> e c', c=self.channels)
        w0_initial_out = rearrange(w0_initial_out, 'e c -> (e c)')

        self.eigenmodel_out = partial_model(
            ensembles = self.num_basis_out * self.channels,
            w0 = w0_out,
            w0_initial = w0_initial_out,
        ).cuda()
        
        # print parameters
        print(f'num params: {sum(p.numel() for p in self.parameters())}')

    # main neural basis compression and reconstruction
    def neural_recon(self, x, model, window_size):
        # make input domain, get eigen-functions
        domain = make_domain(window_size).cuda()
        eigenfunc = model(domain)
        eigenfunc = rearrange(eigenfunc, 't (e c) 1 -> t e c', e=self.num_basis_in, c=self.channels)
        eigenfunc = add_ones(eigenfunc)

        # normalize to have inner product of 1
        eigenfunc = normalize(eigenfunc)
        if x is None: return eigenfunc

        # compute orthonormality (for loss)
        ortho = orthonormality(eigenfunc)

        # get coeffs and reconstructed function
        # ie. this is a compression mechanism
        coeffs = torch.einsum('b t c, t e c -> b e c', x, eigenfunc)
        recon = torch.einsum('b e c, t e c -> b t c', coeffs, eigenfunc)

        return recon, coeffs, eigenfunc, ortho
    
    def set_eigenfuncs(self):
        domain_in = make_domain(self.seq_len).cuda()
        domain_out = make_domain(self.pred_len).cuda()

        eigenfunc_in = self.eigenmodel_in(domain_in)
        eigenfunc_in = rearrange(eigenfunc_in, 't (e c) 1 -> t e c', e=self.num_basis_in, c=self.channels)
        eigenfunc_in = add_ones(eigenfunc_in)
        self.eigenfunc_in = normalize(eigenfunc_in)

        eigenfunc_out = self.eigenmodel_out(domain_out)
        eigenfunc_out = rearrange(eigenfunc_out, 't (e c) 1 -> t e c', e=self.num_basis_out, c=self.channels)
        eigenfunc_out = add_ones(eigenfunc_out)
        self.eigenfunc_out = normalize(eigenfunc_out)

        self.is_train = False
        return

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y=None):
        x = x_enc 
        y = batch_y
        label = x_mark_enc[:, -1]

        if y is not None:
            # get coeffs, perform reconstruction
            recon_in, coeffs_in, eigenfunc_in, ortho_in = self.neural_recon(
                x, self.eigenmodel_in, self.seq_len
            )

            recon_out, coeffs_out, eigenfunc_out, ortho_out = self.neural_recon(
                y, self.eigenmodel_out, self.pred_len
            )

            ortho_loss = ortho_in + ortho_out

            # set eigenfuncs for test time
            self.is_train = True
            self.eigenfunc_in = eigenfunc_in[0]
            self.eigenfunc_out = eigenfunc_out[0]
        else:
            if self.is_train: self.set_eigenfuncs()
            eigenfunc_in = self.eigenfunc_in
            eigenfunc_out = self.eigenfunc_out

            coeffs_in = torch.einsum('b t c, t e c -> b e c', x, eigenfunc_in)

        # predict future coeffs
        coeff_flat = rearrange(coeffs_in, 'b e c -> b (e c)')
        coeff_flat_in = torch.cat([coeff_flat, label], dim=-1)
        coeff_flat_pred = self.linear(coeff_flat_in)
        pred_coeff = rearrange(coeff_flat_pred, 'b (e c) -> b e c', e=self.num_basis_out+1, c=self.channels)

        # reconstruct output function
        pred_recon = torch.einsum('b e c, t e c -> b t c', pred_coeff, eigenfunc_out)

        # losses
        if y is None:
            return pred_recon
        else:
            recon_in_loss = (x - recon_in).square().mean()
            recon_out_loss = (y - recon_out).square().mean()
            pred_coeff_loss = (coeffs_out - pred_coeff).square().mean()
            pred_recon_loss = (y - pred_recon).square().mean()
            loss = recon_in_loss + recon_out_loss + pred_coeff_loss + pred_recon_loss + ortho_loss

            return pred_recon, loss

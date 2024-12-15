from collections import OrderedDict
import torch
from torch._C import _disabled_torch_function_impl
import torch.nn as nn
from torch.nn import init, Module, Conv2d, Linear
import torch.nn.functional as F
from torch.nn.functional import relu, max_pool2d

def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)

def complex_max_pool2d(input,kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):
    '''
    Perform complex max pooling by selecting on the absolute value on the complex values.
    '''
    absolute_value, indices =  max_pool2d(
                               input.abs(), 
                               kernel_size = kernel_size, 
                               stride = stride, 
                               padding = padding, 
                               dilation = dilation,
                               ceil_mode = ceil_mode, 
                               return_indices = True
                            )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresonding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    angle = torch.atan2(input.imag,input.real)
    # get only the phase values selected by max pool
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value \
           * (torch.cos(angle).type(torch.complex64)+1j*torch.sin(angle).type(torch.complex64))


class _ParameterMeta(torch._C._TensorMeta):
    # Make `isinstance(t, Parameter)` return True for custom tensor instances that have the _is_param flag.
    def __instancecheck__(self, instance):
        if self is Parameter:
            if isinstance(instance, torch.Tensor) and getattr(
                instance, "_is_param", False
            ):
                return True
        return super().__instancecheck__(instance)


class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a Parameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead. To use the type as a "
                "Parameter, please correct the detach() semantics defined by "
                "its __torch_dispatch__() implementation."
            )
        t._is_param = True
        return t

    # Note: the 3 methods below only apply to standard Tensor. Parameters of custom tensor types
    # are still considered that custom tensor type and these methods will not be called for them.
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format), self.requires_grad
            )
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()

    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # See Note [Don't serialize hooks]
        hooks = OrderedDict()
        if not state:
            return (
                torch._utils._rebuild_parameter,
                (self.data, self.requires_grad, hooks),
            )

        return (
            torch._utils._rebuild_parameter_with_state,
            (self.data, self.requires_grad, hooks, state),
        )

    __torch_function__ = _disabled_torch_function_impl



class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3)).to(self.device)
            self.bias = Parameter(torch.Tensor(num_features,2)).to(self.device)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype = torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1./n*input.real.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input.imag.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0,2,3])
        else:
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps 
       
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]
                
        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None,:,None,None]*input.real+Rri[None,:,None,None]*input.imag).type(torch.complex64) \
                + 1j*(Rii[None,:,None,None]*input.imag+Rri[None,:,None,None]*input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+\
                    self.bias[None,:,0,None,None]).type(torch.complex64) \
                    +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+\
                    self.bias[None,:,1,None,None]).type(torch.complex64)

        return input

class ComplexConv2d(Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self,input):    
        return apply_complex(self.conv_r, self.conv_i, input)
    
class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ComplexConv2d(3, 10, 5, 1)
        self.bn  = ComplexBatchNorm2d(10)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(500, 500)
        self.fc2 = ComplexLinear(500, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.bn(x)
        x = complex_relu(self.conv2(x))
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1,500)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = x.abs()
        x = F.log_softmax(x, dim=1)
        return x

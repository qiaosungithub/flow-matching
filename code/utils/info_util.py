from utils.logging_util import log_for_0
import utils.state_utils as state_utils
import flax.nnx as nn

# Function to print number of parameters

def nnx_to_dict(p, d: dict):
    for k in p:
        v = p[k]
        # print(type(v))
        if isinstance(v, nn.variablelib.VariableState):
            if v.value is not None:
                d[str(k)] = v.value
        else:
            newd={}
            d[str(k)]=newd
            nnx_to_dict(v, newd)

def print_params(params):
    """This is for nnx version"""
    # convert State to dict
    d={}
    nnx_to_dict(params, d)
    params = d
    # print(params)

    # now reduce to linen version
    params_flatten = state_utils.flatten_state_dict(params)

    # for name, param in params_flatten.items():
    #     # print the mean and std and min and max of the parameter
    #     with open("/kmh-nfs-ssd-eu-mount/code/qiao/papers/DeiT/jax_param.txt", "a") as f:
    #         f.write(f"{name}: {param.mean()}, {param.std()}, {param.min()}, {param.max()}\n")

    total_params = 0
    max_length = max(len(k) for k in params_flatten.keys())
    max_shape = max(len(f"{p.shape}") for p in params_flatten.values())
    max_digits = max(len(f"{p.size:,}") for p in params_flatten.values())
    log_for_0('-' * (max_length + max_digits + max_shape + 8))
    for name, param in params_flatten.items():
        layer_params = param.size
        str_layer_shape = f"{param.shape}".rjust(max_shape) 
        str_layer_params = f"{layer_params:,}".rjust(max_digits)
        log_for_0(f" {name.ljust(max_length)} | {str_layer_shape} | {str_layer_params} ")
        total_params += layer_params
    log_for_0('-' * (max_length + max_digits + max_shape + 8))
    log_for_0(f"Total parameters: {total_params:,}")

# def print_params(params):
#     """This is for linen version"""
#     params_flatten = state_utils.flatten_state_dict(params)

#     # for name, param in params_flatten.items():
#     #     # print the mean and std and min and max of the parameter
#     #     with open("/kmh-nfs-ssd-eu-mount/code/qiao/papers/DeiT/jax_param.txt", "a") as f:
#     #         f.write(f"{name}: {param.mean()}, {param.std()}, {param.min()}, {param.max()}\n")

#     total_params = 0
#     max_length = max(len(k) for k in params_flatten.keys())
#     max_shape = max(len(f"{p.shape}") for p in params_flatten.values())
#     max_digits = max(len(f"{p.size:,}") for p in params_flatten.values())
#     log_for_0('-' * (max_length + max_digits + max_shape + 8))
#     for name, param in params_flatten.items():
#         layer_params = param.size
#         str_layer_shape = f"{param.shape}".rjust(max_shape) 
#         str_layer_params = f"{layer_params:,}".rjust(max_digits)
#         log_for_0(f" {name.ljust(max_length)} | {str_layer_shape} | {str_layer_params} ")
#         total_params += layer_params
#     log_for_0('-' * (max_length + max_digits + max_shape + 8))
#     log_for_0(f"Total parameters: {total_params:,}")

    # exit(114514)
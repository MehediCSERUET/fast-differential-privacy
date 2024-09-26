"""
A large portion of this code is adapted from Opacus v0.15 (https://github.com/pytorch/opacus) 
and from Private-transformers v0.2.3 (https://github.com/lxuechen/private-transformers)
which are licensed under Apache License 2.0.

We have modified it considerably to support book-keeping and BiTFiT.
"""

from typing import Tuple

import torch
import torch.nn as nn
import time

from .supported_layers_grad_samplers import _supported_layers_norm_sample_AND_clipping,_create_or_extend_private_grad

# To store cumulative FLOPS, time, and memory usage
prepare_sample_grad_or_norm_flops = 0
prepare_sample_grad_or_norm_time = 0
prepare_sample_grad_or_norm_memory = 0
MMH = 0

def requires_grad(module: nn.Module) -> bool:
    """
    Checks if any parameters in a specified module require gradients.

    Args:
        module: PyTorch module whose parameters are examined

    Returns:
        Flag indicate if any parameters require gradients
    """
    print("You are now in requires_grad function of autograd_grad_sample file")
    return any(p.initially_requires_grad for p in module.parameters() if hasattr(p,'initially_requires_grad'))


def add_hooks(model: nn.Module, loss_reduction='mean', clipping_mode='MixOpt',bias_only=False,
              clipping_style='all-layer', block_heads=None, named_params=None, named_layers=None,
              clipping_fn=None, numerical_stability_constant=None, max_grad_norm_layerwise=None):
    r"""
    Adds hooks to model to save activations (to layers) and backprop (to params) values.

    The hooks will

    1. save activations into ``layer.activations`` (NOT param.activations) during forward pass.
    Note: BiTFiT is special in that if a layer only requires bias gradient, no need for forward hook
        
    2. compute per-sample grad norm or grad and save in ``param.norm_sample`` or ``param.grad_sample`` during backward pass.

    Args:
        model: Model to which hooks are added.
    """
    #MMH
    print("You are now in add_hooks function of autograd_grad_sample file\nStarting profiling")
    global prepare_sample_grad_or_norm_flops, prepare_sample_grad_or_norm_time, prepare_sample_grad_or_norm_memory, MMH
    # Time tracking
    start_time = time.perf_counter()
    # Start profiling, MMH
    with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof: #MMH
        if hasattr(model, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
    
        handles = []
    
        for name, layer in model.named_modules():
            if type(layer) in _supported_layers_norm_sample_AND_clipping and requires_grad(layer):
                if hasattr(layer.weight,'initially_requires_grad') and layer.weight.initially_requires_grad:
                    #print('Attaching forward hook on', name)
                    handles.append(layer.register_forward_hook(_capture_activations))
                    
                if name in block_heads:
                    def this_backward(this_layer, grad_input, grad_output):
                        _prepare_sample_grad_or_norm(this_layer, grad_output, loss_reduction, clipping_mode,bias_only)
                        _per_block_clip_grad(this_layer, named_params, named_layers, clipping_style, clipping_fn, numerical_stability_constant, max_grad_norm_layerwise)
                else:
                    def this_backward(this_layer, grad_input, grad_output):
                        _prepare_sample_grad_or_norm(this_layer, grad_output, loss_reduction, clipping_mode,bias_only)
    
                # Starting with 1.8.0, can use `register_full_backward_hook`, but slower
                handles.append(layer.register_backward_hook(this_backward))            
        print("You are now in add_hooks function of autograd_grad_sample file")
        model.__dict__.setdefault("autograd_grad_sample_hooks", []).extend(handles)
    
    # End time tracking
    end_time = time.perf_counter()

    # Accumulate FLOPS, time, and memory usage
    prepare_sample_grad_or_norm_flops += sum([event.flops for event in prof.key_averages()])
    prepare_sample_grad_or_norm_time += (end_time - start_time) * 1000  # Convert to milliseconds
    MMH += 1

    # Get memory stats
    if torch.cuda.is_available():
        prepare_sample_grad_or_norm_memory += torch.cuda.memory_allocated()  # Memory in bytes (CUDA)
    else:
        prepare_sample_grad_or_norm_memory += torch.cuda.memory_allocated()  # Use this even for CPU tracking
                  
#MMH
def get_per_block_clip_grad_flops():
    return [prepare_sample_grad_or_norm_flops, prepare_sample_grad_or_norm_time, prepare_sample_grad_or_norm_memory, MMH]

def reset_stats():
    global prepare_sample_grad_or_norm_flops, prepare_sample_grad_or_norm_time, prepare_sample_grad_or_norm_memory
    prepare_sample_grad_or_norm_flops = 0
    prepare_sample_grad_or_norm_time = 0
    prepare_sample_grad_or_norm_memory = 0
    MMH = 0

def remove_hooks(model: nn.Module):
    """Removes hooks added by `add_hooks()`."""
    print("Remove hook function of autograd_grad_sample")
    for handle in model.autograd_grad_sample_hooks:
        handle.remove()
    del model.autograd_grad_sample_hooks


def _capture_activations(layer: nn.Module, inputs: Tuple, outputs: Tuple):
    """Forward hook handler captures AND saves activations."""
    print("Capture_activations function in autograd_grad_sample")
    layer.activations=inputs[0].detach()

def _prepare_sample_grad_or_norm(
    layer: nn.Module,
    grad_output: Tuple[torch.Tensor],
    loss_reduction='mean',
    clipping_mode='MixOpt',
    bias_only=False,
    ):
    """Backward hook handler captures AND saves grad_outputs (book-keeping)."""
    #print("_prepare_sample_grad_or_norm function in autograd_grad_sample file")
        # Start profiling
    with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
        backprops = grad_output[0].detach()
    
        """Computes per-sample grad norm or grad for individual layers."""
        if not hasattr(layer,'activations'):
            layer.activations=None
        if loss_reduction=='mean':
            backprops = backprops * backprops.shape[0] # .backprops should save dL_i/ds, not 1/B*dL_i/ds, the mean reduction is taken care of in privacy engine .step()
        compute_layer_grad_sample, _ = _supported_layers_norm_sample_AND_clipping.get(type(layer))
    
        compute_layer_grad_sample(layer, layer.activations, backprops, clipping_mode)
    
        layer.backprops=backprops
    # Print profiler results
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    #print("FLOPS:", prof.total_average().flops)


def _per_block_clip_grad(
    layer: nn.Module, named_params, named_layers, clipping_style, clipping_fn,
    numerical_stability_constant,max_grad_norm_layerwise
    ):
    #print("_per_block_clip_grad function in autograd_grad_sample file")
    
    # Start profiling
    with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
        if clipping_style not in ['layer-wise','param-wise']:
    
            norm_sample = torch.stack([param.norm_sample for name, param in named_params if hasattr(param,'norm_sample')], dim=0).norm(2, dim=0)
            # compute per-sample grad norm and clipping factor
            if clipping_fn=='automatic':
                C = max_grad_norm_layerwise / (norm_sample + numerical_stability_constant)
            elif clipping_fn=='Abadi':
                C = torch.clamp_max(max_grad_norm_layerwise / (norm_sample + numerical_stability_constant), 1.)
            elif clipping_fn=='global':
                C = (norm_sample<=max_grad_norm_layerwise).float()
            else:
                raise ValueError(f"Unknown clipping function {clipping_fn}. Expected one of Abadi, automatic, global.")
    
            for name, layer in named_layers:
                if hasattr(layer,'weight') and hasattr(layer.weight,'initially_requires_grad') and layer.weight.initially_requires_grad and hasattr(layer,'activations') and hasattr(layer.weight,'norm_sample'):
                    #--- weight, compute clipped gradient
                    _, compute_layer_grad = _supported_layers_norm_sample_AND_clipping.get(type(layer))
                    grad_weight = compute_layer_grad(layer, layer.activations, torch.einsum('b...,b->b...',layer.backprops,C), C)
                    del layer.activations, layer.backprops
                    _create_or_extend_private_grad(layer.weight, grad_weight)
                    
                if hasattr(layer,'bias') and hasattr(layer.bias,'initially_requires_grad') and layer.bias.initially_requires_grad and hasattr(layer.bias,'grad_sample') and hasattr(layer.bias,'norm_sample'):
                    #--- bias, compute clipped gradient
                    grad_bias = torch.einsum("b...,b->...", layer.bias.grad_sample, C)
                    del layer.bias.grad_sample
                    _create_or_extend_private_grad(layer.bias, grad_bias)
                    
        elif clipping_style =='layer-wise':
    
            norm_sample = torch.stack([param.norm_sample for param in layer.parameters() if hasattr(param,'norm_sample')], dim=0).norm(2, dim=0)
            # compute per-sample grad norm and clipping factor
            if clipping_fn=='automatic':
                C = max_grad_norm_layerwise / (norm_sample + numerical_stability_constant)
            elif clipping_fn=='Abadi':
                C = torch.clamp_max(max_grad_norm_layerwise / (norm_sample + numerical_stability_constant), 1.)
            elif clipping_fn=='global':
                C = (norm_sample<=max_grad_norm_layerwise).float()
            else:
                raise ValueError(f"Unknown clipping function {clipping_fn}. Expected one of Abadi, automatic, global.")
        
    
            if hasattr(layer,'weight') and hasattr(layer.weight,'initially_requires_grad') and layer.weight.initially_requires_grad and hasattr(layer,'activations') and hasattr(layer.weight,'norm_sample'):
                #--- weight, compute clipped gradient
                _, compute_layer_grad = _supported_layers_norm_sample_AND_clipping.get(type(layer))
                grad_weight = compute_layer_grad(layer, layer.activations, torch.einsum('b...,b->b...',layer.backprops,C), C)
                del layer.activations, layer.backprops
                if hasattr(layer.weight,'grad_sample'):
                    print(type(layer))
                _create_or_extend_private_grad(layer.weight, grad_weight)
                
            if hasattr(layer,'bias') and hasattr(layer.bias,'initially_requires_grad') and layer.bias.initially_requires_grad and hasattr(layer.bias,'grad_sample') and hasattr(layer.bias,'norm_sample'):
                #--- bias, compute clipped gradient
                grad_bias = torch.einsum("b...,b->...", layer.bias.grad_sample, C)
                del layer.bias.grad_sample
                _create_or_extend_private_grad(layer.bias, grad_bias)
                    
        elif clipping_style=='param-wise':
            if hasattr(layer,'weight') and hasattr(layer.weight,'norm_sample'):
                if clipping_fn=='automatic':
                    C_weight = max_grad_norm_layerwise / (layer.weight.norm_sample + numerical_stability_constant)
                elif clipping_fn=='Abadi':
                    C_weight = torch.clamp_max(max_grad_norm_layerwise / (layer.weight.norm_sample + numerical_stability_constant), 1.)
                elif clipping_fn=='global':
                    C_weight = (layer.weight.norm_sample<=max_grad_norm_layerwise).float()
                else:
                    raise ValueError(f"Unknown clipping function {clipping_fn}. Expected one of Abadi, automatic, global.")
            
            if hasattr(layer,'bias') and hasattr(layer.bias,'norm_sample'):
                if clipping_fn=='automatic':
                    C_bias = max_grad_norm_layerwise / (layer.bias.norm_sample + numerical_stability_constant)
                elif clipping_fn=='Abadi':
                    C_bias = torch.clamp_max(max_grad_norm_layerwise / (layer.bias.norm_sample + numerical_stability_constant), 1.)
                elif clipping_fn=='global':
                    C_bias = (layer.bias.norm_sample<=max_grad_norm_layerwise).float()
                else:
                    raise ValueError(f"Unknown clipping function {clipping_fn}. Expected one of Abadi, automatic, global.")
            
                
            if hasattr(layer,'weight') and hasattr(layer.weight,'initially_requires_grad') and layer.weight.initially_requires_grad and hasattr(layer,'activations') and hasattr(layer.weight,'norm_sample'):
                _, compute_layer_grad = _supported_layers_norm_sample_AND_clipping.get(type(layer))
                grad_weight = compute_layer_grad(layer, layer.activations, torch.einsum('b...,b->b...',layer.backprops,C_weight), C_weight)
                del layer.activations, layer.backprops
                
                _create_or_extend_private_grad(layer.weight, grad_weight)
                
                
            #--- bias, compute clipped gradient
            if hasattr(layer,'bias') and hasattr(layer.bias,'initially_requires_grad') and layer.bias.initially_requires_grad and hasattr(layer.bias,'grad_sample') and hasattr(layer.bias,'norm_sample'):
                grad_bias = torch.einsum("b...,b->...", layer.bias.grad_sample, C_bias)
                del layer.bias.grad_sample
                _create_or_extend_private_grad(layer.bias, grad_bias)
    
        for name,param in named_params:
          if hasattr(param,'norm_sample'):
              del param.norm_sample
    # Print profiler results
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    #print("FLOPS:", prof.total_average().flops)

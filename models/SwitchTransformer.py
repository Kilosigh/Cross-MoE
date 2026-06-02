
import torch
from torch import nn
from typing import List, Dict, Any, Optional, Tuple, Union

from labml_helpers.module import Module
# from labml_nn.transformers.feed_forward import FeedForward
from layers.MyLayers import FeedForward
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.utils import clone_module_list


class SwitchFeedForward(Module):
    """
    ## Routing among multiple FFNs
    """

    def __init__(self, *,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_experts: int,
                 expert: FeedForward,
                 d_in: int,
                 d_out: int,
                 output_routing_distribution: int):
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens
        self.output_routing_distribution = output_routing_distribution
        self.d_in = d_in
        self.d_out = d_out

        # make copies of the FFNs

        self.experts = clone_module_list(expert, n_experts)
        # Routing layer and softmax
        self.switch = nn.Linear(d_in, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[batch_size, seq_len, d_model]`
        """

        # Capture the shape to change shapes later
        batch_size, seq_len, d_model = x.shape
        assert(d_model == self.d_in)
        # Flatten the sequence and batch dimensions
        x = x.view(-1, d_model)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.switch(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        # route_prob_max[routes[i]] corresponds to the maximum of route_prob[i]
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Get indexes of tokens going to each expert, 
        # [torch.eq(routes, i) for i in range(self.n_experts)][i] 是第i个专家所分到的token的位图表示
        # [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)][i]则表示第i个专家分到的token的index列表 
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(batch_size * seq_len, self.d_out)

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[batch_size, seq_len, d_model]`
        final_output = final_output.view(batch_size, seq_len, self.d_out)

        total = counts.sum(dim=-1, keepdims=True)
        route_frac = counts / total
        aux_loss = self.n_experts * (route_frac * route_prob.sum(0)).sum()
        
        route_prob = route_prob.reshape(batch_size, seq_len, -1)
        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        # return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max
        if self.output_routing_distribution:
            return final_output, aux_loss, route_prob
        return final_output, aux_loss


class LayerConfigBuilder:
    """
    Helper class to build layer configurations more easily
    """
    
    @staticmethod
    def create_uniform_config(num_layers: int, 
                            base_config: Dict[str, Any], 
                            d_in: int,
                            d_model: int,
                            drop_out: float,
                            d_out_progression: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Create uniform configuration for all layers
        
        Args:
            num_layers: Number of MoE layers
            base_config: Base configuration to use for all layers
            d_model: Input dimension
            d_out_progression: List of output dimensions for each layer.
                             If None, all layers will have d_out = d_model
        """
        if d_out_progression is None:
            d_out_progression = [d_model] * num_layers
        
        d_out_progression[0] = d_in
        
        assert len(d_out_progression) == num_layers, \
            "d_out_progression length must match num_layers"
        
        configs = []
        config = base_config.copy()
        config['d_out'] = d_model
        config["expert"] = FeedForward(d_in, d_model, drop_out)
        configs.append(config)

        for i in range(num_layers - 1):
            config = base_config.copy()
            config['d_out'] = d_model
            config["expert"] = FeedForward(d_model, d_model, drop_out)
            configs.append(config)
        
        return configs
    
    @staticmethod
    def create_pyramid_config(num_layers: int,
                            base_config: Dict[str, Any],
                            d_model: int,
                            expert_reduction_factor: float = 0.5,
                            dim_reduction_factor: float = 1.0) -> List[Dict[str, Any]]:
        """
        Create pyramid-like configuration where later layers have fewer experts
        
        Args:
            num_layers: Number of MoE layers
            base_config: Base configuration
            d_model: Input dimension
            expert_reduction_factor: Factor to reduce experts each layer
            dim_reduction_factor: Factor to reduce dimensions each layer
        """
        configs = []
        current_experts = base_config['n_experts']
        current_dim = d_model
        
        for i in range(num_layers):
            config = base_config.copy()
            config['n_experts'] = max(1, int(current_experts))
            config['d_out'] = max(1, int(current_dim))
            configs.append(config)
            
            current_experts *= expert_reduction_factor
            current_dim *= dim_reduction_factor
        
        return configs

class MultiLayerSwitchFeedForward(nn.Module):
    """
    ## Multi-Layer MoE with router-experts-router-experts structure
    """
    
    def __init__(self, 
                 layer_configs: List[Dict[str, Any]],
                 d_model: int,
                 output_all_routing_distributions: bool = False):
        """
        Args:
            layer_configs: List of configuration dictionaries for each MoE layer
                Each config should contain:
                - capacity_factor: float
                - drop_tokens: bool  
                - is_scale_prob: bool
                - n_experts: int
                - expert: nn.Module (the expert architecture)
                - d_out: int (output dimension for this layer)
                - output_routing_distribution: bool
            d_model: Input dimension for the first layer
            output_all_routing_distributions: Whether to return routing distributions from all layers
        """
        super().__init__()
        
        self.num_layers = len(layer_configs)
        self.output_all_routing_distributions = output_all_routing_distributions
        self.d_model = d_model
        
        # Build the MoE layers
        self.moe_layers = nn.ModuleList()
        current_d_in = d_model
        
        for i, config in enumerate(layer_configs):
            # Set input dimension for current layer
            layer_config = config.copy()
            layer_config['d_in'] = current_d_in
            
            # Create the MoE layer
            print("s3")
            print(type(layer_config["expert"]))
            print(type(layer_config["n_experts"]))
            moe_layer = SwitchFeedForward(**layer_config)
            self.moe_layers.append(moe_layer)
            
            # Update input dimension for next layer
            current_d_in = config['d_out']
        
        self.final_d_out = current_d_in
    
    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                               Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            If output_all_routing_distributions is False:
                (final_output, total_aux_loss)
            If output_all_routing_distributions is True:
                (final_output, total_aux_loss, all_routing_distributions)
        """
        current_input = x
        total_aux_loss = 0.0
        all_routing_distributions = []
        
        for i, moe_layer in enumerate(self.moe_layers):
            if moe_layer.output_routing_distribution:
                output, aux_loss, route_prob = moe_layer(current_input)
                all_routing_distributions.append(route_prob)
            else:
                output, aux_loss = moe_layer(current_input)
                all_routing_distributions.append(None)
            
            current_input = output
            total_aux_loss = total_aux_loss + aux_loss
        
        if self.output_all_routing_distributions:
            return current_input, total_aux_loss, all_routing_distributions
        else:
            return current_input, total_aux_loss


class MultiLayerSwitchFeedForwardWithResidual(nn.Module):
    """
    ## Multi-Layer MoE with router-experts-router-experts structure and residual connections
    """
    
    def __init__(self, 
                 layer_configs: List[Dict[str, Any]],
                 d_model: int,
                 output_all_routing_distributions: bool = False,
                 use_layer_norm: bool = True,
                 dropout: float = 0.1,
                 residual_scaling: float = 1.0):
        """
        Args:
            layer_configs: List of configuration dictionaries for each MoE layer
                Each config should contain:
                - capacity_factor: float
                - drop_tokens: bool  
                - is_scale_prob: bool
                - n_experts: int
                - expert: nn.Module (the expert architecture)
                - d_out: int (output dimension for this layer)
                - output_routing_distribution: bool
            d_model: Input dimension for the first layer
            output_all_routing_distributions: Whether to return routing distributions from all layers
            use_layer_norm: Whether to apply layer normalization before each MoE layer
            dropout: Dropout rate for residual connections
            residual_scaling: Scaling factor for residual connections
        """
        super().__init__()
        
        self.num_layers = len(layer_configs)
        self.output_all_routing_distributions = output_all_routing_distributions
        self.d_model = d_model
        self.use_layer_norm = use_layer_norm
        self.dropout = nn.Dropout(dropout)
        self.residual_scaling = residual_scaling
        
        # Build the MoE layers
        self.moe_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        current_d_in = d_model
        
        for i, config in enumerate(layer_configs):
            # Set input dimension for current layer
            layer_config = config.copy()
            layer_config['d_in'] = current_d_in
            
            # Create the MoE layer
            moe_layer = SwitchFeedForward(**layer_config)
            self.moe_layers.append(moe_layer)
            
            # Create projection layer if dimensions don't match for residual connection
            if current_d_in != config['d_out']:
                projection = nn.Linear(current_d_in, config['d_out'])
            else:
                projection = nn.Identity()
                
            self.projection_layers.append(projection)
            
            # Create layer normalization if enabled
            if use_layer_norm:
                layer_norm = nn.LayerNorm(current_d_in)
                self.layer_norms.append(layer_norm)
            
            # Update input dimension for next layer
            current_d_in = config['d_out']
        
        self.final_d_out = current_d_in
    
    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                               Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            If output_all_routing_distributions is False:
                (final_output, total_aux_loss)
            If output_all_routing_distributions is True:
                (final_output, total_aux_loss, all_routing_distributions)
        """
        current_input = x
        total_aux_loss = 0.0
        all_routing_distributions = []
        
        for i, (moe_layer, projection_layer) in enumerate(zip(self.moe_layers, self.projection_layers)):
            # Store residual connection input
            residual = current_input
            
            # Apply layer normalization before MoE layer (Pre-LN architecture)
            if self.use_layer_norm:
                normalized_input = self.layer_norms[i](current_input)
            else:
                normalized_input = current_input
            
            # Forward through MoE layer
            if moe_layer.output_routing_distribution:
                moe_output, aux_loss, route_prob = moe_layer(normalized_input)
                all_routing_distributions.append(route_prob)
            else:
                moe_output, aux_loss = moe_layer(normalized_input)
                all_routing_distributions.append(None)
            
            # Apply dropout to MoE output
            moe_output = self.dropout(moe_output)
            
            # Project residual if dimensions don't match
            projected_residual = projection_layer(residual)
            
            # Add residual connection with scaling
            current_input = projected_residual + self.residual_scaling * moe_output
            
            # Accumulate auxiliary loss
            total_aux_loss = total_aux_loss + aux_loss
        
        if self.output_all_routing_distributions:
            return current_input, total_aux_loss, all_routing_distributions
        else:
            return current_input, total_aux_loss

class SwitchTransformerLayer(Module):
    """
    # Switch Transformer Block

    This is the same as [normal transformer block](../models.html#TransformerLayer)
    with handling extra outputs of switch feedforward module.
    """

    def __init__(self, *,
                 d_model: int,
                 attn: MultiHeadAttention,
                 feed_forward: SwitchFeedForward,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `attn` is the attention module
        * `feed_forward` is the feed forward module (which is the switching module in this case)
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the switching feed-forward network
        ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x, counts, route_prob, n_dropped, route_prob_max


class SwitchTransformer(Module):
    """
    ## Switch Transformer
    """

    def __init__(self, layer: SwitchTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Run through each transformer layer
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []
        for layer in self.layers:
            x, f, p, n_d, p_max = layer(x=x, mask=mask)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)
        # Finally, normalize the vectors
        x = self.norm(x)
        #
        return x, torch.stack(counts), torch.stack(route_prob), n_dropped, torch.stack(route_prob_max)


# see https://arxiv.org/pdf/2203.15556 Appendix F
def calculate_transformer_flops(
    seq_len: int,
    vocab_size: int,
    d_model: int,
    key_size: int,
    num_heads: int,
    ffw_size: int,
    num_layers: int,
) -> dict:
    """
    Calculates flops required for one step with one batch size
    Args:
        seq_len: Sequence length
        vocab_size: Vocabulary size
        d_model: Model dimension
        key_size: Key dimension
        num_heads: Number of attention heads
        ffw_size: Feed-forward layer size
        num_layers: Number of transformer layers
    """

    # Embeddings
    embedding_flops = 2 * seq_len * vocab_size * d_model

    # Single Attention Layer
    key_query_value_proj = 2 * 3 * seq_len * d_model * (key_size * num_heads)
    key_query_logits = 2 * seq_len * seq_len * (key_size * num_heads)
    softmax_ops = 3 * num_heads * seq_len * seq_len
    softmax_query_reduction = 2 * seq_len * seq_len * (key_size * num_heads)
    final_linear = 2 * seq_len * (key_size * num_heads) * d_model

    total_attention_flops = (
        key_query_value_proj
        + key_query_logits
        + softmax_ops
        + softmax_query_reduction
        + final_linear
    )

    # Single Dense Block
    dense_block_flops = 2 * seq_len * (d_model * ffw_size + d_model * ffw_size)

    # Final Logits
    final_logits_flops = 2 * seq_len * d_model * vocab_size

    # Total forward pass
    total_forward_pass = (
        embedding_flops
        + num_layers * (total_attention_flops + dense_block_flops)
        + final_logits_flops
    )

    # Backward pass is approximately 2x forward pass
    total_backward_pass = 2 * total_forward_pass

    # Total forward + backward
    total_flops = total_forward_pass + total_backward_pass

    return total_flops

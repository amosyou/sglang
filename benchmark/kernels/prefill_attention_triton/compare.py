from typing import Tuple

import numpy as np
import torch
import triton
from fastmoe_mla import context_mla_fwd

# Import both implementations
from sglang_mla import extend_attention_fwd


def generate_test_data(
    batch_size: int = 4,
    seq_len: int = 128,
    n_heads_q: int = 8,
    n_heads_kv: int = 4,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, ...]:
    """Generate random test data for both attention implementations."""
    device = torch.device("cuda")

    # Generate sequence lengths
    b_seq_len = torch.randint(
        seq_len // 2, seq_len, (batch_size,), dtype=torch.int32, device=device
    )
    b_seq_len_prefix = torch.randint(
        1, seq_len // 4, (batch_size,), dtype=torch.int32, device=device
    )
    b_seq_len_extend = b_seq_len - b_seq_len_prefix

    # Calculate total tokens and other required values
    total_tokens = torch.sum(b_seq_len).item()
    extend_tokens = torch.sum(b_seq_len_extend).item()
    max_len = torch.max(b_seq_len).item()

    # Generate start locations
    b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)

    b_start_loc_extend = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    # Generate request to tokens mapping
    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    req_to_tokens = torch.empty((batch_size, max_len), dtype=torch.int32, device=device)
    for i in range(batch_size):
        req_to_tokens[i, : b_seq_len[i]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len[i]
        )

    # Generate random data for k_buffer and v_buffer
    k_buffer = torch.empty(
        (total_tokens, n_heads_kv, 576), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.02)
    v_buffer = torch.empty(
        (total_tokens, n_heads_kv, 512), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.02)

    # Generate data for extend tensors
    k_extend = torch.empty((extend_tokens, n_heads_kv, 576), dtype=dtype, device=device)
    v_extend = torch.empty((extend_tokens, n_heads_kv, 512), dtype=dtype, device=device)
    q_extend = torch.empty(
        (extend_tokens, n_heads_q, 576), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.02)

    # Copy relevant portions to extend tensors
    for i in range(batch_size):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]

        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]

    # Generate q_buffer for second implementation
    q_buffer = torch.empty((total_tokens, n_heads_q, 576), dtype=dtype, device=device)

    pt = 0
    for i in range(batch_size):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    return (
        q_extend,
        k_extend,
        v_extend,
        q_buffer,
        k_buffer,
        v_buffer,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_seq_len_prefix,
        b_start_loc_extend,
        b_seq_len_extend,
        max_len,
    )


def compare_outputs(
    o1: torch.Tensor, o2: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-3
) -> Tuple[float, float, bool]:
    """Compare two output tensors and return statistics."""
    diff = torch.abs(o1 - o2)
    mean_diff = torch.mean(diff).item()
    max_diff = torch.max(diff).item()
    is_close = torch.allclose(o1, o2, rtol=rtol, atol=atol)
    return mean_diff, max_diff, is_close


def run_test_with_retry(config: dict, max_retries: int = 3) -> None:
    """Try running test with progressively smaller configs if needed."""
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]

    for attempt in range(max_retries):
        try:
            run_test(batch_size=batch_size, seq_len=seq_len)
            return
        except triton.runtime.errors.OutOfResources as e:
            print(f"\nAttempt {attempt + 1} failed with configuration:")
            print(f"batch_size={batch_size}, seq_len={seq_len}")
            print(f"Error: {e}")

            # Reduce sizes for next attempt
            batch_size = max(1, batch_size - 1)
            seq_len = seq_len // 2

            if attempt < max_retries - 1:
                print(f"\nRetrying with smaller configuration:")
                print(f"batch_size={batch_size}, seq_len={seq_len}")
            else:
                print("\nFailed all retry attempts")
                raise


def run_test(
    batch_size: int = 4,
    seq_len: int = 128,
    n_heads_q: int = 8,
    n_heads_kv: int = 4,
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> None:
    """Run both implementations and compare results."""
    torch.manual_seed(seed)

    print(f"\nRunning test with parameters:")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Query heads: {n_heads_q}")
    print(f"Key/Value heads: {n_heads_kv}")
    print(f"Data type: {dtype}")

    # Generate test data
    (
        q_extend,
        k_extend,
        v_extend,
        q_buffer,
        k_buffer,
        v_buffer,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_seq_len_prefix,
        b_start_loc_extend,
        b_seq_len_extend,
        max_len,
    ) = generate_test_data(batch_size, seq_len, n_heads_q, n_heads_kv, dtype)

    # Initialize output tensors
    extend_tokens = q_extend.shape[0]
    o_extend = torch.empty((extend_tokens, n_heads_q, 512), dtype=dtype, device="cuda")
    o_buffer = torch.empty(
        (k_buffer.shape[0], n_heads_q, 512), dtype=dtype, device="cuda"
    )

    # Run second implementation
    print("running fastmoe mla")
    context_mla_fwd(
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len
    )

    print("running sglang mla")
    # Run first implementation
    extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_seq_len_prefix,
        b_start_loc_extend,
        b_seq_len_extend,
        max_len,
        torch.max(b_seq_len_extend).item(),
    )

    # Extract relevant portions from o_buffer for comparison
    o_mla = torch.empty_like(o_extend)
    pt = 0
    for i in range(batch_size):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_mla[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend

    # Compare outputs
    mean_diff, max_diff, is_close = compare_outputs(o_extend, o_mla)

    print("\nResults:")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Outputs match within tolerance: {is_close}")


if __name__ == "__main__":
    # Test with different configurations
    # Start with smaller configs for testing
    configs = [
        {"batch_size": 2, "seq_len": 16},
        # {"batch_size": 4, "seq_len": 32},
        # {"batch_size": 2, "seq_len": 64},
    ]

    for config in configs:
        try:
            run_test_with_retry(config)
        except Exception as e:
            print(f"Failed config {config}: {e}")
            continue

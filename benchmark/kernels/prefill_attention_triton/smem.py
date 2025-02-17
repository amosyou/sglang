import torch
import triton


def calculate_shared_memory(
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_QK_DIM: int,
    BLOCK_V_DIM: int,
    dtype_size: int = 4,  # float32 by default
):
    """Calculate shared memory requirements for context attention."""

    # Memory for Q chunk (M x QK_DIM)
    q_memory = BLOCK_M * BLOCK_QK_DIM * dtype_size

    # Memory for K chunk (QK_DIM x N)
    k_memory = BLOCK_QK_DIM * BLOCK_N * dtype_size

    # Memory for V chunk (N x V_DIM)
    v_memory = BLOCK_N * BLOCK_V_DIM * dtype_size

    # Memory for QK accumulator (M x N)
    qk_acc_memory = BLOCK_M * BLOCK_N * dtype_size

    # Memory for output accumulator (M x V_DIM)
    acc_memory = BLOCK_M * BLOCK_V_DIM * dtype_size

    total_memory = q_memory + k_memory + v_memory + qk_acc_memory + acc_memory

    print(f"\nShared Memory Analysis:")
    print(f"Q chunk memory: {q_memory:,} bytes")
    print(f"K chunk memory: {k_memory:,} bytes")
    print(f"V chunk memory: {v_memory:,} bytes")
    print(f"QK accumulator memory: {qk_acc_memory:,} bytes")
    print(f"Output accumulator memory: {acc_memory:,} bytes")
    print(f"Total estimated memory: {total_memory:,} bytes")

    return total_memory


def print_gpu_info():
    """Print GPU capabilities and shared memory limits."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"\nGPU Information:")
        print(f"Device: {props.name}")
        print(f"CUDA Capability: {props.major}.{props.minor}")
        # print(f"Shared Memory per Block: {props.shared_memory_per_multiprocessor:,} bytes")
        # print(f"Max Shared Memory per Block: {props.max_shared_memory_per_multiprocessor:,} bytes")
    else:
        print("No GPU available")


def analyze_kernel_memory(
    batch_size: int = 2,
    seq_len: int = 64,
    head_dim: int = 576,
    v_dim: int = 512,
    dtype: torch.dtype = torch.float16,
):
    """Analyze memory requirements for different configurations."""

    print_gpu_info()

    # Get block sizes based on head dimension
    BLOCK = 16  # Base block size from the implementation

    if head_dim == 576:
        BLOCK_QK_DIM = 512  # Process in chunks
        QK_FULL_DIM = 576
    else:
        BLOCK_QK_DIM = head_dim
        QK_FULL_DIM = head_dim

    BLOCK_V_DIM = v_dim

    dtype_size = torch.finfo(dtype).bits // 8

    print(f"\nConfiguration:")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Head dimension: {head_dim}")
    print(f"Value dimension: {v_dim}")
    print(f"Data type: {dtype} ({dtype_size} bytes)")
    print(
        f"Block sizes: M={BLOCK}, N={BLOCK}, QK_DIM={BLOCK_QK_DIM}, V_DIM={BLOCK_V_DIM}"
    )

    mem_required = calculate_shared_memory(
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_QK_DIM=BLOCK_QK_DIM,
        BLOCK_V_DIM=BLOCK_V_DIM,
        dtype_size=dtype_size,
    )

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        # if mem_required > props.shared_memory_per_multiprocessor:
        #     print(f"\nWARNING: Required memory ({mem_required:,} bytes) exceeds ")
        #     print(f"available shared memory ({props.shared_memory_per_multiprocessor:,} bytes)")
        #     print("Consider reducing block sizes or using smaller data type")


if __name__ == "__main__":
    # Test different configurations
    configs = [
        {"batch_size": 2, "seq_len": 64},
        {"batch_size": 4, "seq_len": 128},
        {"batch_size": 8, "seq_len": 256},
    ]

    for config in configs:
        print("\n" + "=" * 50)
        print(f"Testing configuration: {config}")
        analyze_kernel_memory(**config)

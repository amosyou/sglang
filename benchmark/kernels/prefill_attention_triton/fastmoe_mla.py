import torch
import triton
import triton.language as tl

CUDA_CAPABILITY = torch.cuda.get_device_capability()
cached_kernel = None


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_QK_DIM: tl.constexpr,
    BLOCK_V_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QK_FULL_DIM: tl.constexpr,  # Full dimension (e.g., 576)
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_qk = tl.arange(0, BLOCK_QK_DIM)  # Power of 2 size (e.g., 512)
    offs_v = tl.arange(0, BLOCK_V_DIM)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Initialize accumulators for the full attention computation
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_V_DIM], dtype=tl.float32)

    # Main sequence length loop
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Initialize accumulator for QK across dimension chunks
        qk_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Process in power-of-2 sized chunks for the inner dimension
        for dim_offset in range(0, QK_FULL_DIM, BLOCK_QK_DIM):
            # Adjust offsets for current chunk
            off_q = (
                (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
                + cur_head * stride_qh
                + offs_qk[None, :]
                + dim_offset
            )
            off_k = (
                offs_n[None, :] * stride_kbs
                + cur_kv_head * stride_kh
                + offs_qk[:, None]
                + dim_offset
            )

            # Load Q and K chunks with proper masking
            q_mask = (offs_m[:, None] < cur_batch_seq_len) & (
                offs_qk[None, :] + dim_offset < QK_FULL_DIM
            )
            k_mask = ((start_n + offs_n[None, :]) < cur_batch_seq_len) & (
                offs_qk[:, None] + dim_offset < QK_FULL_DIM
            )

            q = tl.load(Q + off_q, mask=q_mask, other=0.0)
            k = tl.load(
                K + off_k + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                mask=k_mask,
                other=0.0,
            )

            # Accumulate partial QK dot products
            qk_acc += tl.dot(q, k)

        # After accumulating all dimension chunks, apply scale and causal mask
        qk_acc *= sm_scale
        qk_acc = tl.where(
            offs_m[:, None] >= (start_n + offs_n[None, :]), qk_acc, float("-inf")
        )

        # Compute local attention statistics
        m_ij = tl.max(qk_acc, 1)
        p = tl.exp(qk_acc - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Update running statistics
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # Update attention probabilities
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        if start_n > 0:
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]

        # Load and accumulate V
        off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_v[None, :]
        v = tl.load(
            V + off_v + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        # Update running statistics for next iteration
        l_i = l_i_new
        m_i = m_i_new

    # Store final output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_v[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


def context_mla_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    Lq, Lk, Lv, Lo = q.shape[-1], k.shape[-1], v.shape[-1], o.shape[-1]

    assert Lq == Lk, "Q and K dimensions must match"
    assert Lv == Lo, "V and Output dimensions must match"

    # Find the largest power of 2 that's less than Lq
    if Lq == 576:
        BLOCK_QK_DIM = 512
        QK_FULL_DIM = 576
    else:
        BLOCK_QK_DIM = Lq
        QK_FULL_DIM = Lq
    BLOCK_V_DIM = Lv
    assert Lq <= 576 and Lv <= 512, "Dimensions too large"

    BLOCK = 16  # For T4 compatibility

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_QK_DIM=BLOCK_QK_DIM,  # Process in chunks of 512
        BLOCK_V_DIM=BLOCK_V_DIM,
        BLOCK_N=BLOCK,
        QK_FULL_DIM=QK_FULL_DIM,  # Full dimension we want to process
        num_warps=num_warps,
        num_stages=1,
    )

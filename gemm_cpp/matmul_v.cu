constexpr float COEFF = 0.867325070f;
#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define OFFSET(row, col, stride) ((row) * (stride) + (col))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v(int M, int N, int K, float *A, float *B, float *C)
{
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;
    int num_shared_block = CEIL_DIV(K, BK);

    __shared__ float As[2][BK][BM]; // transpose shared A for avoid bank conflict, for double buffering
    __shared__ float Bs[2][BK][BN]; // for double buffering

    float accum[TM][TN] = {0.};

    const int load_a_cache_time = (BK * BM) / thread_num / 4; // Each thread load 4 float
    const int load_b_cache_time = (BK * BN) / thread_num / 4; // Each thread load 4 float

    float load_a_cache[4 * load_a_cache_time];
    //    float load_a_cache[4];
    //    float load_b_cache[4 * load_b_cache_time];

    A = &A[OFFSET(blockIdx.y * BM, 0, K)]; // Set block start position
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int a_tile_row = thread_id / (BK / 4);
    int a_tile_col = thread_id % (BK / 4) * 4;
    int a_tile_stride = BM / load_a_cache_time;
    //    printf("A tile row, col, stride %d, %d, %d", a_tile_row, a_tile_col, a_tile_stride);

    int b_tile_row = thread_id / (BN / 4);
    int b_tile_col = thread_id % (BN / 4) * 4;
    int b_tile_stride = BK / load_b_cache_time;

    float As_cache[2][TM] = {0.}; // double buffering
    float Bs_cache[2][TN] = {0.}; // double buffering

    int write_idx = 0;

#pragma unroll
    for (int i = 0; i < num_shared_block; ++i)
    {
#pragma unroll
        for (int m = 0; m < BM; m += a_tile_stride)
        {
            int cache_idx = m / a_tile_stride * 4;
            FETCH_FLOAT4(load_a_cache[cache_idx]) =
                FETCH_FLOAT4(A[OFFSET(a_tile_row + m, a_tile_col, K)]);
            // Use load_a_cache for load 4 float at a time
            // As is saved as transpose matrix
            As[write_idx][a_tile_col][a_tile_row + m] = load_a_cache[cache_idx];
            // 这里 stride = 128，有 shared memory bank 冲突
            As[write_idx][a_tile_col + 1][a_tile_row + m] = load_a_cache[cache_idx + 1];
            As[write_idx][a_tile_col + 2][a_tile_row + m] = load_a_cache[cache_idx + 2];
            As[write_idx][a_tile_col + 3][a_tile_row + m] = load_a_cache[cache_idx + 3];
        }
#pragma unroll
        for (int k = 0; k < BK; k += b_tile_stride)
        {
            FETCH_FLOAT4(Bs[write_idx][b_tile_row + k][b_tile_col]) =
                FETCH_FLOAT4(B[OFFSET(b_tile_row + k, b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;     // Start position of next tile block to be processed
        B += BK * N; // Start position of next tile block to be processed

#pragma unroll
        for (int k = 0; k < BK; ++k)
        {
#pragma unroll
            for (int m = 0; m < TM; m += 4)
            {
                int A_row = threadIdx.y * TM + m;
                FETCH_FLOAT4(As_cache[write_idx][m]) = FETCH_FLOAT4(As[write_idx][k][A_row]);
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4)
            {
                int B_col = threadIdx.x * TN + n;
                FETCH_FLOAT4(Bs_cache[write_idx][n]) = FETCH_FLOAT4(Bs[write_idx][k][B_col]);
            }
#pragma unroll
            for (int m = 0; m < TM; ++m)
            {
#pragma unroll
                for (int n = 0; n < TN; ++n)
                {
                    accum[m][n] += expf(-0.5f * (As_cache[write_idx][m] + Bs_cache[write_idx][n]) * (As_cache[write_idx][m] + Bs_cache[write_idx][n]));
                }
            }
        }
        write_idx ^= 1;
    }

#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
        int C_row = threadIdx.y * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4)
        {
            int C_col = threadIdx.x * TN + n;
            // FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]);
            load_a_cache[0] = accum[m][n];
            load_a_cache[1] = accum[m][n + 1];
            load_a_cache[2] = accum[m][n + 2];
            load_a_cache[3] = accum[m][n + 3];
            FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]) = FETCH_FLOAT4(load_a_cache);
        }
    }
}

void gemm_v_launcher(int batch_size, int in_feats, int out_feats,
                     float *x, float *bias, float *result)
{
    const int size = 8;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(out_feats, BN), CEIL_DIV(batch_size, BM)); // note: change M and N here
    gemm_v<BM, BN, BK, TM, TN>
        <<<grid, block>>>(batch_size, out_feats, in_feats, x, bias, result);
}

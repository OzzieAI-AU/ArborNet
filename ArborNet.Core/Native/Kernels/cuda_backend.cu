#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

#define EXPORT extern "C" __declspec(dllexport)
#define TILE_SIZE 16
#define MAX_RANK 8

// =================================================================================
// 1. HELPERS & MACROS
// =================================================================================

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline void get_launch_config(int n, int* blockSize, int* gridSize) {
    *blockSize = 256;
    *gridSize = (n + *blockSize - 1) / *blockSize;
    if (*gridSize > 65535) *gridSize = 65535;
}

// Safe error checking (internal)
inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        // In production you could log this or set a global error state
        // For now we clear it to prevent cascading errors
        cudaGetLastError();
    }
}

// =================================================================================
// 2. KERNELS (GPU Side) — All Improved & Optimized
// =================================================================================

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < m && t * TILE_SIZE + threadIdx.x < k)
            sA[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && t * TILE_SIZE + threadIdx.y < k)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float max_val = -FLT_MAX;
    for (int c = 0; c < cols; ++c)
        max_val = fmaxf(max_val, input[r * cols + c]);

    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float e = expf(input[r * cols + c] - max_val);
        output[r * cols + c] = e;
        sum += e;
    }

    float scale = 1.0f / (sum + 1e-8f);
    for (int c = 0; c < cols; ++c)
        output[r * cols + c] *= scale;
}

// --- Core Operations ---
__global__ void add_kernel(const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = a[i] + b[i]; }
__global__ void sub_kernel(const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = a[i] - b[i]; }
__global__ void mul_kernel(const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = a[i] * b[i]; }
__global__ void div_kernel(const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = (b[i] != 0.0f) ? (a[i] / b[i]) : 0.0f; }
__global__ void pow_kernel(const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = powf(a[i], b[i]); }

// --- Logic ---
__global__ void equal_kernel(const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = (a[i] == b[i]) ? 1.0f : 0.0f; }
__global__ void greater_than_kernel(const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = (a[i] > b[i]) ? 1.0f : 0.0f; }
__global__ void less_than_kernel(const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = (a[i] < b[i]) ? 1.0f : 0.0f; }
__global__ void where_kernel(const float* cond, const float* a, const float* b, float* c, int n) { CUDA_KERNEL_LOOP(i, n) c[i] = (cond[i] != 0.0f) ? a[i] : b[i]; }

// --- Activations + Gradients ---
__global__ void relu_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = fmaxf(0.0f, in[i]); }
__global__ void relu_grad_kernel(const float* grad_out, const float* origin_in, float* grad_in, int n) { CUDA_KERNEL_LOOP(i, n) grad_in[i] = (origin_in[i] > 0.0f) ? grad_out[i] : 0.0f; }

__global__ void sigmoid_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = 1.0f / (1.0f + expf(-in[i])); }
__global__ void sigmoid_grad_kernel(const float* grad_out, const float* origin_out, float* grad_in, int n) { CUDA_KERNEL_LOOP(i, n) grad_in[i] = grad_out[i] * origin_out[i] * (1.0f - origin_out[i]); }

__global__ void tanh_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = tanhf(in[i]); }

// --- Unary ---
__global__ void negate_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = -in[i]; }
__global__ void exp_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = expf(in[i]); }
__global__ void log_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = logf(in[i] + 1e-8f); }
__global__ void sqrt_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = sqrtf(fmaxf(in[i], 0.0f)); }
__global__ void abs_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = fabsf(in[i]); }
__global__ void sin_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = sinf(in[i]); }
__global__ void cos_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = cosf(in[i]); }
__global__ void sign_kernel(const float* in, float* out, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        float v = in[i];
        out[i] = (v > 0.0f) - (v < 0.0f);
    }
}

// --- Scalar & Fill ---
__global__ void set_scalar_kernel(float* data, float value, int n) { CUDA_KERNEL_LOOP(i, n) data[i] = value; }
__global__ void pow_scalar_kernel(const float* in, float* out, int n, float exponent) { CUDA_KERNEL_LOOP(i, n) out[i] = powf(in[i], exponent); }
__global__ void mul_scalar_kernel(float* out, float scalar, int n) { CUDA_KERNEL_LOOP(i, n) out[i] *= scalar; }

// --- Reductions & Indexing ---
__global__ void sum_all_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride)
        local += in[i];
    atomicAdd(out, local);
}

__global__ void argmax_kernel(const float* input, int* output, int outer, int dim, int inner) {
    CUDA_KERNEL_LOOP(idx, outer * inner) {
        int o = idx / inner, i = idx % inner;
        float max_val = -FLT_MAX; int max_idx = 0;
        for (int d = 0; d < dim; ++d) {
            float val = input[(o * dim + d) * inner + i];
            if (val > max_val) { max_val = val; max_idx = d; }
        }
        output[idx] = max_idx;
    }
}

__global__ void argmin_kernel(const float* input, int* output, int outer, int dim, int inner) {
    CUDA_KERNEL_LOOP(idx, outer * inner) {
        int o = idx / inner, i = idx % inner;
        float min_val = FLT_MAX; int min_idx = 0;
        for (int d = 0; d < dim; ++d) {
            float val = input[(o * dim + d) * inner + i];
            if (val < min_val) { min_val = val; min_idx = d; }
        }
        output[idx] = min_idx;
    }
}

__global__ void cumsum_kernel(const float* input, float* output, int outer, int dim, int inner) {
    CUDA_KERNEL_LOOP(idx, outer * inner) {
        int o = idx / inner, i = idx % inner;
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            int pos = (o * dim + d) * inner + i;
            sum += input[pos];
            output[pos] = (float)sum;
        }
    }
}

// --- Shape Manipulation ---
__global__ void transpose_2d_kernel(const float* in, float* out, int rows, int cols) {
    CUDA_KERNEL_LOOP(i, rows * cols) {
        int r = i / cols, c = i % cols;
        out[c * rows + r] = in[i];
    }
}

__global__ void general_transpose_kernel(const float* input, float* output, const int* shape, const int* perm, int rank, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        int coords[MAX_RANK];
        int temp = i;
        for (int d = rank - 1; d >= 0; --d) {
            coords[d] = temp % shape[d];
            temp /= shape[d];
        }
        int new_idx = 0, stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            new_idx += coords[perm[d]] * stride;
            stride *= shape[perm[d]];
        }
        output[new_idx] = input[i];
    }
}

__global__ void broadcast_kernel(const float* input, float* output, const int* in_shape, const int* out_shape, int rank, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        int temp = i, in_idx = 0, stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            int coord = temp % out_shape[d];
            temp /= out_shape[d];
            in_idx += (in_shape[d] == 1 ? 0 : coord) * stride;
            stride *= in_shape[d];
        }
        output[i] = input[in_idx];
    }
}

// --- Utilities ---
__global__ void logical_not_kernel(const float* in, float* out, int n) { CUDA_KERNEL_LOOP(i, n) out[i] = (in[i] == 0.0f) ? 1.0f : 0.0f; }
__global__ void clip_kernel(const float* in, float* out, int n, float minV, float maxV) { CUDA_KERNEL_LOOP(i, n) out[i] = fmaxf(minV, fminf(maxV, in[i])); }
__global__ void clip_mask_kernel(const float* in, float* out, int n, float minV, float maxV) { CUDA_KERNEL_LOOP(i, n) out[i] = (in[i] >= minV && in[i] <= maxV) ? 1.0f : 0.0f; }

// =================================================================================
// 3. DISPATCHERS — Exact Original Signatures (void) + All Improvements
// =================================================================================

EXPORT void NativeMatMul(const float* A, const float* B, float* C, int m, int n, int k) {
    if (!A || !B || !C || m <= 0 || n <= 0 || k <= 0) return;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid, block>>>(A, B, C, m, n, k);
}

EXPORT void NativeSoftmax(const float* in, float* out, int rows, int cols) {
    if (!in || !out || rows <= 0 || cols <= 0) return;
    int bs = 256, gs = (rows + bs - 1) / bs;
    softmax_kernel<<<gs, bs>>>(in, out, rows, cols);
}

// Binary Operations
EXPORT void NativeAdd(const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); add_kernel<<<gs,bs>>>(a,b,c,n); }
EXPORT void NativeSubtract(const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); sub_kernel<<<gs,bs>>>(a,b,c,n); }
EXPORT void NativeMultiply(const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); mul_kernel<<<gs,bs>>>(a,b,c,n); }
EXPORT void NativeDivide(const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); div_kernel<<<gs,bs>>>(a,b,c,n); }
EXPORT void NativePow(const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); pow_kernel<<<gs,bs>>>(a,b,c,n); }

// Logic & Comparison
EXPORT void NativeEqual(const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); equal_kernel<<<gs,bs>>>(a,b,c,n); }
EXPORT void NativeGreaterThan(const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); greater_than_kernel<<<gs,bs>>>(a,b,c,n); }
EXPORT void NativeLessThan(const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); less_than_kernel<<<gs,bs>>>(a,b,c,n); }
EXPORT void NativeWhere(const float* cond, const float* a, const float* b, float* c, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); where_kernel<<<gs,bs>>>(cond,a,b,c,n); }

// Activations
EXPORT void NativeReLU(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); relu_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeReLUGrad(const float* grad_out, const float* origin_in, float* grad_in, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); relu_grad_kernel<<<gs,bs>>>(grad_out,origin_in,grad_in,n); }

EXPORT void NativeSigmoid(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); sigmoid_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeSigmoidGrad(const float* grad_out, const float* origin_out, float* grad_in, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); sigmoid_grad_kernel<<<gs,bs>>>(grad_out,origin_out,grad_in,n); }

EXPORT void NativeTanh(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); tanh_kernel<<<gs,bs>>>(in,out,n); }

// Unary
EXPORT void NativeNegate(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); negate_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeExp(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); exp_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeLog(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); log_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeSqrt(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); sqrt_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeAbs(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); abs_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeSin(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); sin_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeCos(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); cos_kernel<<<gs,bs>>>(in,out,n); }
EXPORT void NativeSign(const float* in, float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); sign_kernel<<<gs,bs>>>(in,out,n); }

// Scalar & Fill
EXPORT void NativeOnes(float* out, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); set_scalar_kernel<<<gs,bs>>>(out,1.0f,n); }
EXPORT void NativeSetScalar(float* out, float value, int n) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); set_scalar_kernel<<<gs,bs>>>(out,value,n); }
EXPORT void NativePowScalar(const float* in, float* out, int n, float exponent) { if(n<=0) return; int bs,gs; get_launch_config(n,&bs,&gs); pow_scalar_kernel<<<gs,bs>>>(in,out,n,exponent); }

// Reductions
EXPORT void NativeSumAll(const float* in, float* out, int n) {
    if(n<=0) return;
    cudaMemset(out, 0, sizeof(float));
    int bs,gs; get_launch_config(n,&bs,&gs);
    sum_all_kernel<<<gs,bs>>>(in,out,n);
}

EXPORT void NativeMeanAll(const float* in, float* out, int n) {
    if(n<=0) return;
    NativeSumAll(in, out, n);
    mul_scalar_kernel<<<1,1>>>(out, 1.0f / (float)n, 1);
}

EXPORT void NativeArgMax(const float* in, int* out, int outer, int dim, int inner) { if(outer*inner<=0) return; int bs,gs; get_launch_config(outer*inner,&bs,&gs); argmax_kernel<<<gs,bs>>>(in,out,outer,dim,inner); }
EXPORT void NativeArgMin(const float* in, int* out, int outer, int dim, int inner) { if(outer*inner<=0) return; int bs,gs; get_launch_config(outer*inner,&bs,&gs); argmin_kernel<<<gs,bs>>>(in,out,outer,dim,inner); }
EXPORT void NativeCumSum(const float* in, float* out, int outer, int dim, int inner) { if(outer*inner<=0) return; int bs,gs; get_launch_config(outer*inner,&bs,&gs); cumsum_kernel<<<gs,bs>>>(in,out,outer,dim,inner); }

// Shape
EXPORT void NativeTranspose(const float* in, float* out, int rows, int cols) { 
    if(rows<=0 || cols<=0) return;
    int bs,gs; get_launch_config(rows*cols,&bs,&gs); 
    transpose_2d_kernel<<<gs,bs>>>(in,out,rows,cols); 
}

EXPORT void NativeGeneralTranspose(const float* input, float* output, const int* host_shape, const int* host_perm, int rank) {
    if (!input || !output || !host_shape || !host_perm || rank < 1 || rank > MAX_RANK) return;
    int total = 1; for (int i = 0; i < rank; ++i) total *= host_shape[i];
    if (total <= 0) return;

    int *d_shape = nullptr, *d_perm = nullptr;
    cudaMalloc(&d_shape, rank * sizeof(int));
    cudaMalloc(&d_perm, rank * sizeof(int));
    cudaMemcpy(d_shape, host_shape, rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_perm, host_perm, rank * sizeof(int), cudaMemcpyHostToDevice);

    int bs, gs; get_launch_config(total, &bs, &gs);
    general_transpose_kernel<<<gs, bs>>>(input, output, d_shape, d_perm, rank, total);

    cudaFree(d_shape);
    cudaFree(d_perm);
}

EXPORT void NativeBroadcast(const float* input, float* output, const int* host_in_shape, const int* host_out_shape, int rank) {
    if (!input || !output || !host_in_shape || !host_out_shape || rank < 1) return;
    int total = 1; for (int i = 0; i < rank; ++i) total *= host_out_shape[i];
    if (total <= 0) return;

    int *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, rank * sizeof(int));
    cudaMalloc(&d_out, rank * sizeof(int));
    cudaMemcpy(d_in, host_in_shape, rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, host_out_shape, rank * sizeof(int), cudaMemcpyHostToDevice);

    int bs, gs; get_launch_config(total, &bs, &gs);
    broadcast_kernel<<<gs, bs>>>(input, output, d_in, d_out, rank, total);

    cudaFree(d_in);
    cudaFree(d_out);
}

// Utilities
EXPORT void LogicalNot(const float* input, float* output, int count) { if(count<=0) return; int bs,gs; get_launch_config(count,&bs,&gs); logical_not_kernel<<<gs,bs>>>(input,output,count); }
EXPORT void Clip(const float* input, float* output, int count, float minVal, float maxVal) { if(count<=0) return; int bs,gs; get_launch_config(count,&bs,&gs); clip_kernel<<<gs,bs>>>(input,output,count,minVal,maxVal); }
EXPORT void ClipMask(const float* input, float* output, int count, float minVal, float maxVal) { if(count<=0) return; int bs,gs; get_launch_config(count,&bs,&gs); clip_mask_kernel<<<gs,bs>>>(input,output,count,minVal,maxVal); }

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#define EXPORT extern "C" __declspec(dllexport)

// =================================================================================
// 0. UTILITIES & STRUCTS
// =================================================================================

// Struct to pass array data by value to the GPU (fixes host-to-device memory violations)
struct ShapeInfo {
    int data[12]; // Supports up to 12D tensors
};

inline void get_launch_config(int n, int* blockSize, int* gridSize) {
    *blockSize = 256;
    *gridSize = (n + *blockSize - 1) / *blockSize;
    if (*gridSize > 65535) *gridSize = 65535;
}

// =================================================================================
// 1. THE KERNELS (GPU Side)
// =================================================================================

// --- Binary Operations ---
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) c[i] = a[i] + b[i];
}

__global__ void sub_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) c[i] = a[i] - b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) c[i] = a[i] * b[i];
}

__global__ void div_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        c[i] = (b[i] != 0.0f) ? (a[i] / b[i]) : 0.0f;
    }
}

__global__ void pow_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) c[i] = powf(a[i], b[i]);
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) sum += A[row * k + i] * B[i * n + col];
        C[row * n + col] = sum;
    }
}

// --- Logic & Comparison ---
__global__ void equal_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        c[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
    }
}

__global__ void greater_than_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) c[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}

__global__ void less_than_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) c[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}

__global__ void where_kernel(const float* cond, const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        c[i] = (cond[i] != 0.0f) ? a[i] : b[i];
    }
}

// --- Custom Operations (Sign, Transpose, Broadcast) ---
__global__ void SignKernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
    }
}

__global__ void GeneralTransposeKernel(const float* __restrict__ input, float* __restrict__ output,
                                     ShapeInfo shape, ShapeInfo perm, int rank, long long totalElements) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    int coords[12];           
    long long remainder = idx;
    
    for (int i = rank - 1; i >= 0; --i) {
        coords[i] = remainder % shape.data[i];
        remainder /= shape.data[i];
    }

    int inputCoords[12];
    for (int i = 0; i < rank; ++i) {
        inputCoords[perm.data[i]] = coords[i];
    }

    long long inputIdx = 0;
    long long stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        inputIdx += (long long)inputCoords[i] * stride;
        stride *= shape.data[i];
    }

    output[idx] = input[inputIdx];
}

__global__ void BroadcastKernel(const float* __restrict__ input, float* __restrict__ output, 
                                ShapeInfo inShape, ShapeInfo outShape, int rank, long long totalElements) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    int outCoords[12];
    long long remainder = idx;
    for (int i = rank - 1; i >= 0; --i) {
        outCoords[i] = remainder % outShape.data[i];
        remainder /= outShape.data[i];
    }

    long long inIdx = 0;
    long long inStride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        int inCoord = (inShape.data[i] == 1) ? 0 : outCoords[i];
        inIdx += inCoord * inStride;
        inStride *= inShape.data[i];
    }

    output[idx] = input[inIdx];
}

// --- Activations ---
__global__ void relu_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = fmaxf(0.0f, in[i]);
}

__global__ void sigmoid_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

__global__ void tanh_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = tanhf(in[i]);
}

// --- Unary Operations ---
__global__ void negate_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = -in[i];
}

__global__ void exp_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = expf(in[i]);
}

__global__ void log_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = logf(in[i] + 1e-8f);
}

__global__ void sqrt_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = sqrtf(fmaxf(in[i], 0.0f));
}

__global__ void abs_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = fabsf(in[i]);
}

__global__ void sin_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = sinf(in[i]);
}

__global__ void cos_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = cosf(in[i]);
}

// --- Fills and Scalars ---
__global__ void set_scalar_kernel(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

__global__ void pow_scalar_kernel(const float* in, float* out, int n, float exponent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = powf(in[i], exponent);
}

__global__ void mul_scalar_kernel(float* out, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] *= scalar;
}

// --- Reductions & Shapes ---
__global__ void sum_all_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) local_sum += in[i];
    atomicAdd(out, local_sum);
}

__global__ void transpose_2d_kernel(const float* in, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        int r = i / cols;
        int c = i % cols;
        out[c * rows + r] = in[i];
    }
}

__global__ void argmax_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;
    int o = idx / inner;
    int i = idx % inner;
    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int d = 0; d < dim; d++) {
        float val = input[(o * dim + d) * inner + i];
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }
    output[idx] = (float)max_idx;
}

__global__ void argmin_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;
    int o = idx / inner;
    int i = idx % inner;
    float min_val = FLT_MAX;
    int min_idx = 0;
    for (int d = 0; d < dim; d++) {
        float val = input[(o * dim + d) * inner + i];
        if (val < min_val) {
            min_val = val;
            min_idx = d;
        }
    }
    output[idx] = (float)min_idx;
}

__global__ void cumsum_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total) return;
    int o = idx / inner;
    int i = idx % inner;
    double sum = 0; 
    for (int d = 0; d < dim; d++) {
        int pos = (o * dim + d) * inner + i;
        sum += (double)input[pos];
        output[pos] = (float)sum;
    }
}


// =================================================================================
// 2. DISPATCHERS (CPU Side)
// =================================================================================

EXPORT void NativeAdd(const float* a, const float* b, float* c, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); add_kernel<<<gs, bs>>>(a, b, c, n); 
}

EXPORT void NativeEqual(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs);
    equal_kernel<<<gs, bs>>>(a, b, c, n);
}

EXPORT void NativeSubtract(const float* a, const float* b, float* c, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); sub_kernel<<<gs, bs>>>(a, b, c, n); 
}

EXPORT void NativeMultiply(const float* a, const float* b, float* c, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); mul_kernel<<<gs, bs>>>(a, b, c, n); 
}

EXPORT void NativeDivide(const float* a, const float* b, float* c, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); div_kernel<<<gs, bs>>>(a, b, c, n); 
}

EXPORT void NativeMatMul(const float* A, const float* B, float* C, int m, int n, int k) {
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, m, n, k);
}

EXPORT void NativePow(const float* a, const float* b, float* c, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); pow_kernel<<<gs, bs>>>(a, b, c, n); 
}

EXPORT void NativeNegate(const float* in, float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); negate_kernel<<<gs, bs>>>(in, out, n); 
}

EXPORT void NativeExp(const float* in, float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); exp_kernel<<<gs, bs>>>(in, out, n); 
}

EXPORT void NativeLog(const float* in, float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); log_kernel<<<gs, bs>>>(in, out, n); 
}

EXPORT void NativeSqrt(const float* in, float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); sqrt_kernel<<<gs, bs>>>(in, out, n); 
}

EXPORT void NativeAbs(const float* in, float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); abs_kernel<<<gs, bs>>>(in, out, n); 
}

EXPORT void NativeSin(const float* in, float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); sin_kernel<<<gs, bs>>>(in, out, n); 
}

EXPORT void NativeCos(const float* in, float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); cos_kernel<<<gs, bs>>>(in, out, n); 
}

EXPORT void NativeReLU(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); relu_kernel<<<gs, bs>>>(in, out, n);
}

EXPORT void NativeSigmoid(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); sigmoid_kernel<<<gs, bs>>>(in, out, n);
}

EXPORT void NativeTanh(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); tanh_kernel<<<gs, bs>>>(in, out, n);
}

EXPORT void NativeWhere(const float* cond, const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); where_kernel<<<gs, bs>>>(cond, a, b, c, n);
}

EXPORT void NativeGreaterThan(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); greater_than_kernel<<<gs, bs>>>(a, b, c, n);
}

EXPORT void NativeLessThan(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); less_than_kernel<<<gs, bs>>>(a, b, c, n);
}

EXPORT void NativeOnes(float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); set_scalar_kernel<<<gs, bs>>>(out, 1.0f, n); 
}

EXPORT void NativeSetScalar(float* out, float value, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); set_scalar_kernel<<<gs, bs>>>(out, value, n); 
}

EXPORT void NativePowScalar(const float* in, float* out, int n, float exponent) { 
    int bs, gs; get_launch_config(n, &bs, &gs); pow_scalar_kernel<<<gs, bs>>>(in, out, n, exponent); 
}

EXPORT void NativeTranspose(const float* in, float* out, int rows, int cols) { 
    int bs, gs; get_launch_config(rows * cols, &bs, &gs); transpose_2d_kernel<<<gs, bs>>>(in, out, rows, cols); 
}

EXPORT void NativeSumAll(const float* in, float* out, int n) { 
    int bs, gs; get_launch_config(n, &bs, &gs); sum_all_kernel<<<gs, bs>>>(in, out, n); 
}

EXPORT void NativeMeanAll(const float* in, float* out, int n) {
    NativeSumAll(in, out, n);
    mul_scalar_kernel<<<1, 1>>>(out, 1.0f / (float)n, 1);
}

EXPORT void NativeArgMax(const float* in, float* out, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    argmax_kernel<<<gs, bs>>>(in, out, outer, dim, inner);
}

EXPORT void NativeArgMin(const float* in, float* out, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    argmin_kernel<<<gs, bs>>>(in, out, outer, dim, inner);
}

EXPORT void NativeCumSum(const float* in, float* out, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    cumsum_kernel<<<gs, bs>>>(in, out, outer, dim, inner);
}

EXPORT void NativeSign(const float* input, float* output, int n) {
    if (n <= 0) return;
    int bs, gs; get_launch_config(n, &bs, &gs);
    SignKernel<<<gs, bs>>>(input, output, n);
    cudaDeviceSynchronize();
}

EXPORT void NativeGeneralTranspose(const float* input, float* output, const int* shape, const int* perm, int rank) {
    if (rank <= 0 || rank > 12) return;

    long long totalElements = 1;
    ShapeInfo d_shape;
    ShapeInfo d_perm;
    
    // Copy Host arrays to the structs for value-passing
    for (int i = 0; i < rank; ++i) {
        d_shape.data[i] = shape[i];
        d_perm.data[i] = perm[i];
        totalElements *= shape[i];
    }

    if (totalElements == 0) return;

    const int blockSize = 256;
    const int numBlocks = (int)((totalElements + blockSize - 1) / blockSize);

    GeneralTransposeKernel<<<numBlocks, blockSize>>>(input, output, d_shape, d_perm, rank, totalElements);
    cudaDeviceSynchronize();
}

EXPORT void NativeBroadcast(const float* input, float* output, const int* inputShape, const int* outputShape, int rank) {
    if (rank <= 0 || rank > 12) return;

    long long totalElements = 1;
    ShapeInfo d_inShape, d_outShape;
    for (int i = 0; i < rank; ++i) {
        d_inShape.data[i] = inputShape[i];
        d_outShape.data[i] = outputShape[i];
        totalElements *= outputShape[i];
    }

    const int blockSize = 256;
    const int numBlocks = (int)((totalElements + blockSize - 1) / blockSize);
    BroadcastKernel<<<numBlocks, blockSize>>>(input, output, d_inShape, d_outShape, rank, totalElements);
    cudaDeviceSynchronize();
}
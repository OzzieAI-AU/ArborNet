#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#define EXPORT extern "C" __declspec(dllexport)

// =================================================================================
// 0. UTILITIES, STRUCTS & COMPLEX SYSTEM
// =================================================================================

struct ShapeInfo {
    int data[12];
};

struct SliceInfo {
    int data[12];
};

struct ComplexDouble {
    double real;
    double imag;
};

__device__ inline ComplexDouble complex_add(ComplexDouble a, ComplexDouble b) {
    return { a.real + b.real, a.imag + b.imag };
}

__device__ inline ComplexDouble complex_mul(ComplexDouble a, ComplexDouble b) {
    return { a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real };
}

__device__ inline ComplexDouble complex_tanh(ComplexDouble z) {
    double x = z.real;
    double y = z.imag;
    double tx = tanh(x);
    double ty = tan(y);
    double denom = 1.0 + tx * tx * ty * ty;
    if (denom == 0.0 || isnan(denom) || isinf(denom)) return { 0.0, 0.0 };
    return { (tx * (1.0 + ty * ty)) / denom, (ty * (1.0 - tx * tx)) / denom };
}

// Optimized global launch configuration helper
inline void get_launch_config(int n, int* blockSize, int* gridSize) {
    *blockSize = 256;
    *gridSize = (n + *blockSize - 1) / *blockSize;
}

// =================================================================================
// 1. CUDA KERNELS
// =================================================================================

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
        float val = b[i];
        c[i] = (val != 0.0f) ? (a[i] / val) : 0.0f;
    }
}

__global__ void pow_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) c[i] = powf(a[i], b[i]);
}

__global__ void equal_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        c[i] = (fabsf(a[i] - b[i]) < 1e-6f) ? 1.0f : 0.0f;
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

__global__ void sign_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
    }
}

__global__ void set_scalar_kernel(float* data, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) data[i] = value;
}

__global__ void pow_scalar_kernel(const float* in, float* out, int n, float exponent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] = powf(in[i], exponent);
}

__global__ void mul_scalar_kernel(float* out, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) out[i] *= scalar;
}

__global__ void transpose_2d_kernel(const float* in, float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int r = i / cols;
        int c = i % cols;
        out[c * rows + r] = in[i];
    }
}

// FIXED: Correct multidimensional transposition addressing the fatal out-of-bounds bug
__global__ void GeneralTransposeKernel(const float* __restrict__ input, float* __restrict__ output,
    ShapeInfo shape, ShapeInfo outShape, ShapeInfo perm, int rank, long long totalElements) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    int outCoords[12];
    long long remainder = idx;

    // Decompose output thread index using the output dimensions
    for (int i = rank - 1; i >= 0; --i) {
        outCoords[i] = remainder % outShape.data[i];
        remainder /= outShape.data[i];
    }

    // Remap coordinates back to the original input mapping rules
    int inCoords[12];
    for (int i = 0; i < rank; ++i) {
        inCoords[perm.data[i]] = outCoords[i];
    }

    // Convert input coordinates to a flat input index
    long long inputIdx = 0;
    long long stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        inputIdx += (long long)inCoords[i] * stride;
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

__global__ void slice_kernel(const float* input, float* output,
    SliceInfo inShape, SliceInfo outShape,
    SliceInfo starts, SliceInfo steps,
    int rank, long long totalElements) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    int outCoords[12];
    long long remainder = idx;
    for (int i = rank - 1; i >= 0; --i) {
        outCoords[i] = remainder % outShape.data[i];
        remainder /= outShape.data[i];
    }

    long long inIdx = 0;
    long long stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        int inCoord = starts.data[i] + outCoords[i] * steps.data[i];
        inIdx += (long long)inCoord * stride;
        stride *= inShape.data[i];
    }

    output[idx] = input[inIdx];
}

__global__ void slice_grad_kernel(const float* gradOut, float* gradIn,
    SliceInfo originalShape, SliceInfo newShape,
    SliceInfo starts, SliceInfo steps,
    int rank, long long totalElements) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    int outCoords[12];
    long long remainder = idx;
    for (int i = rank - 1; i >= 0; --i) {
        outCoords[i] = remainder % newShape.data[i];
        remainder /= newShape.data[i];
    }

    long long inIdx = 0;
    long long stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        int inCoord = starts.data[i] + outCoords[i] * steps.data[i];
        inIdx += (long long)inCoord * stride;
        stride *= originalShape.data[i];
    }

    atomicAdd(&gradIn[inIdx], gradOut[idx]);
}

__global__ void concat_copy_kernel(const float* input, float* output,
    int outerSize, int inputConcatSize,
    int totalConcatSize, int innerSize,
    int currentOffset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = outerSize * inputConcatSize * innerSize;
    for (int i = idx; i < totalElements; i += blockDim.x * gridDim.x) {
        int o = i / (inputConcatSize * innerSize);
        int rem = i % (inputConcatSize * innerSize);
        int c = rem / innerSize;
        int in_val = rem % innerSize;

        int outIdx = o * (totalConcatSize * innerSize) + (currentOffset + c) * innerSize + in_val;
        output[outIdx] = input[i];
    }
}

__global__ void gather_kernel(const float* input, const float* indices, float* output, int batch, int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < batch; i += blockDim.x * gridDim.x) {
        int classIdx = (int)indices[i];
        if (classIdx >= 0 && classIdx < classes) {
            output[i] = input[i * classes + classIdx];
        }
        else {
            output[i] = 0.0f;
        }
    }
}

__global__ void gather_grad_kernel(const float* gradOut, const float* indices, float* gradIn, int batch, int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < batch; i += blockDim.x * gridDim.x) {
        int classIdx = (int)indices[i];
        if (classIdx >= 0 && classIdx < classes) {
            atomicAdd(&gradIn[i * classes + classIdx], gradOut[i]);
        }
    }
}

// FIXED: High-performance Shared Memory reduction replacing global atomic collisions
__global__ void sum_all_reduction_kernel(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    while (i < n) {
        local_sum += in[i];
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Loop-unrolled reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

__global__ void mean_axis_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int o = i / inner;
        int in_val = i % inner;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += input[(o * dim + d) * inner + in_val];
        }
        output[i] = sum / (float)dim;
    }
}

__global__ void argmax_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int o = i / inner;
        int in_val = i % inner;
        float max_val = -FLT_MAX;
        int max_idx = 0;
        for (int d = 0; d < dim; d++) {
            float val = input[(o * dim + d) * inner + in_val];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }
        output[i] = (float)max_idx;
    }
}

__global__ void argmin_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int o = i / inner;
        int in_val = i % inner;
        float min_val = FLT_MAX;
        int min_idx = 0;
        for (int d = 0; d < dim; d++) {
            float val = input[(o * dim + d) * inner + in_val];
            if (val < min_val) {
                min_val = val;
                min_idx = d;
            }
        }
        output[i] = (float)min_idx;
    }
}

__global__ void max_axis_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int o = i / inner;
        int in_val = i % inner;
        float max_val = -FLT_MAX;
        for (int d = 0; d < dim; d++) {
            float val = input[(o * dim + d) * inner + in_val];
            if (val > max_val) max_val = val;
        }
        output[i] = max_val;
    }
}

__global__ void min_axis_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int o = i / inner;
        int in_val = i % inner;
        float min_val = FLT_MAX;
        for (int d = 0; d < dim; d++) {
            float val = input[(o * dim + d) * inner + in_val];
            if (val < min_val) min_val = val;
        }
        output[i] = min_val;
    }
}

__global__ void cumsum_kernel(const float* input, float* output, int outer, int dim, int inner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int o = i / inner;
        int in_val = i % inner;
        double sum = 0;
        for (int d = 0; d < dim; d++) {
            int pos = (o * dim + d) * inner + in_val;
            sum += (double)input[pos];
            output[pos] = (float)sum;
        }
    }
}

// FIXED: Highly optimized 2D Shared Memory Tiling Matrix Multiplication (16x16 tiles)
#define TILE_DIM 16
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float sum = 0.0f;

    for (int q = 0; q < (k + TILE_DIM - 1) / TILE_DIM; ++q) {
        if (row < m && (q * TILE_DIM + tx) < k)
            s_A[ty][tx] = A[row * k + q * TILE_DIM + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (col < n && (q * TILE_DIM + ty) < k)
            s_B[ty][tx] = B[(q * TILE_DIM + ty) * n + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int j = 0; j < TILE_DIM; ++j) {
            sum += s_A[ty][j] * s_B[j][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; r < rows; r += stride) {
        const float* inRow = input + r * cols;
        float* outRow = output + r * cols;

        float maxVal = -FLT_MAX;
        for (int i = 0; i < cols; i++) {
            if (inRow[i] > maxVal) maxVal = inRow[i];
        }

        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            float e = expf(inRow[i] - maxVal);
            outRow[i] = e;
            sum += e;
        }

        for (int i = 0; i < cols; i++) {
            outRow[i] /= sum;
        }
    }
}

__global__ void relu_grad_kernel(const float* gradOut, const float* originIn, float* gradIn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        gradIn[i] = (originIn[i] > 0.0f) ? gradOut[i] : 0.0f;
    }
}

__global__ void sigmoid_grad_kernel(const float* gradOut, const float* originOut, float* gradIn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        float s = originOut[i];
        gradIn[i] = gradOut[i] * s * (1.0f - s);
    }
}

__global__ void logical_not_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        output[i] = (input[i] == 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void clip_kernel(const float* input, float* output, int n, float minVal, float maxVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        output[i] = fminf(fmaxf(input[i], minVal), maxVal);
    }
}

__global__ void clip_mask_kernel(const float* input, float* output, int n, float minVal, float maxVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x * gridDim.x; i < n; i += stride) {
        output[i] = (input[i] >= minVal && input[i] <= maxVal) ? 1.0f : 0.0f;
    }
}

__global__ void holonomic_kernel(const ComplexDouble* inputs, const ComplexDouble* weights,
    const ComplexDouble* intWeights, ComplexDouble* outputs,
    int inputSize, int neuronCount, int fractalDepth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int n = idx; n < neuronCount; n += blockDim.x * gridDim.x) {
        ComplexDouble psi = { 0.0, 0.0 };
        for (int i = 0; i < inputSize; i++) {
            ComplexDouble prod = complex_mul(inputs[i], weights[n * inputSize + i]);
            psi = complex_add(psi, prod);
        }

        ComplexDouble z = { 0.0, 0.0 };
        for (int t = 0; t < fractalDepth; t++) {
            ComplexDouble term = complex_mul(intWeights[n], z);
            z = complex_tanh(complex_add(term, psi));
        }

        outputs[n] = z;
    }
}

// =================================================================================
// 2. DISPATCHERS (CPU Side Wrappers for P/Invoke)
// NOTE: Removed all blocking system synchronization blockages (cudaDeviceSynchronize)
// =================================================================================

EXPORT void NativeAdd(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); add_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativeSubtract(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); sub_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativeMultiply(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); mul_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativeDivide(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); div_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativePow(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); pow_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativePowTensor(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); pow_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativeEqual(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); equal_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativeGreaterThan(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); greater_than_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativeLessThan(const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); less_than_kernel << <gs, bs >> > (a, b, c, n);
}

EXPORT void NativeWhere(const float* cond, const float* a, const float* b, float* c, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); where_kernel << <gs, bs >> > (cond, a, b, c, n);
}

EXPORT void NativeReLU(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); relu_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeReLUGrad(const float* gradOut, const float* originIn, float* gradIn, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); relu_grad_kernel << <gs, bs >> > (gradOut, originIn, gradIn, n);
}

EXPORT void NativeSigmoid(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); sigmoid_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeSigmoidGrad(const float* gradOut, const float* originOut, float* gradIn, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); sigmoid_grad_kernel << <gs, bs >> > (gradOut, originOut, gradIn, n);
}

EXPORT void NativeTanh(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); tanh_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeNegate(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); negate_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeExp(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); exp_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeLog(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); log_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeSqrt(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); sqrt_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeAbs(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); abs_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeSin(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); sin_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeCos(const float* in, float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); cos_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeSign(const float* input, float* output, int n) {
    if (n <= 0) return;
    int bs, gs; get_launch_config(n, &bs, &gs);
    sign_kernel << <gs, bs >> > (input, output, n);
}

EXPORT void NativeOnes(float* out, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); set_scalar_kernel << <gs, bs >> > (out, 1.0f, n);
}

EXPORT void NativeSetScalar(float* out, float value, int n) {
    int bs, gs; get_launch_config(n, &bs, &gs); set_scalar_kernel << <gs, bs >> > (out, value, n);
}

EXPORT void NativePowScalar(const float* in, float* out, int n, float exponent) {
    int bs, gs; get_launch_config(n, &bs, &gs); pow_scalar_kernel << <gs, bs >> > (in, out, n, exponent);
}

EXPORT void NativeTranspose(const float* in, float* out, int rows, int cols) {
    int bs, gs; get_launch_config(rows * cols, &bs, &gs); transpose_2d_kernel << <gs, bs >> > (in, out, rows, cols);
}

EXPORT void NativeSumAll(const float* in, float* out, int n) {
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    // ZERO native global output buffer to correctly initialize sum state
    cudaMemset(out, 0, sizeof(float));
    sum_all_reduction_kernel << <gs, bs >> > (in, out, n);
}

EXPORT void NativeMeanAll(const float* in, float* out, int n) {
    NativeSumAll(in, out, n);
    mul_scalar_kernel << <1, 1 >> > (out, 1.0f / (float)n, 1);
}

EXPORT void NativeArgMax(const float* in, float* out, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    argmax_kernel << <gs, bs >> > (in, out, outer, dim, inner);
}

EXPORT void NativeArgMin(const float* in, float* out, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    argmin_kernel << <gs, bs >> > (in, out, outer, dim, inner);
}

EXPORT void NativeCumSum(const float* in, float* out, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    cumsum_kernel << <gs, bs >> > (in, out, outer, dim, inner);
}

EXPORT void NativeGeneralTranspose(const float* input, float* output, const int* shape, const int* perm, int rank) {
    if (rank <= 0 || rank > 12) return;

    long long totalElements = 1;
    ShapeInfo d_shape, d_outShape, d_perm;

    for (int i = 0; i < rank; ++i) {
        d_shape.data[i] = shape[i];
        d_outShape.data[i] = shape[perm[i]]; // Calculate output shape using transpose layout rules
        d_perm.data[i] = perm[i];
        totalElements *= shape[i];
    }

    if (totalElements == 0) return;

    const int blockSize = 256;
    const int numBlocks = (int)((totalElements + blockSize - 1) / blockSize);

    GeneralTransposeKernel << <numBlocks, blockSize >> > (input, output, d_shape, d_outShape, d_perm, rank, totalElements);
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
    BroadcastKernel << <numBlocks, blockSize >> > (input, output, d_inShape, d_outShape, rank, totalElements);
}

EXPORT void NativeMatMul(const float* A, const float* B, float* C, int m, int n, int k) {
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
    matmul_tiled_kernel << <gridSize, blockSize >> > (A, B, C, m, n, k);
}

EXPORT void NativeSoftmax(const float* input, float* output, int rows, int cols) {
    int bs, gs; get_launch_config(rows, &bs, &gs);
    softmax_kernel << <gs, bs >> > (input, output, rows, cols);
}

EXPORT void NativeLogicalNot(const float* input, float* output, int count) {
    int bs, gs; get_launch_config(count, &bs, &gs);
    logical_not_kernel << <gs, bs >> > (input, output, count);
}

EXPORT void NativeClip(const float* input, float* output, int count, float minVal, float maxVal) {
    int bs, gs; get_launch_config(count, &bs, &gs);
    clip_kernel << <gs, bs >> > (input, output, count, minVal, maxVal);
}

EXPORT void NativeClipMask(const float* input, float* output, int count, float minVal, float maxVal) {
    int bs, gs; get_launch_config(count, &bs, &gs);
    clip_mask_kernel << <gs, bs >> > (input, output, count, minVal, maxVal);
}

EXPORT void NativeMeanAxis(const float* input, float* output, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    mean_axis_kernel << <gs, bs >> > (input, output, outer, dim, inner);
}

EXPORT void NativeMaxAxis(const float* input, float* output, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    max_axis_kernel << <gs, bs >> > (input, output, outer, dim, inner);
}

EXPORT void NativeMinAxis(const float* input, float* output, int outer, int dim, int inner) {
    int bs, gs; get_launch_config(outer * inner, &bs, &gs);
    min_axis_kernel << <gs, bs >> > (input, output, outer, dim, inner);
}

EXPORT void NativeGather(const float* input, const float* indices, float* output, int batch, int classes) {
    int bs, gs; get_launch_config(batch, &bs, &gs);
    gather_kernel << <gs, bs >> > (input, indices, output, batch, classes);
}

EXPORT void NativeGatherGrad(const float* gradOut, const float* indices, float* gradIn, int batch, int classes) {
    int bs, gs; get_launch_config(batch, &bs, &gs);
    gather_grad_kernel << <gs, bs >> > (gradOut, indices, gradIn, batch, classes);
}

EXPORT void NativeSlice(const float* input, float* output,
    const int* inShape, const int* outShape,
    const int* starts, const int* steps, int rank) {
    if (rank <= 0 || rank > 12) return;
    long long totalElements = 1;
    SliceInfo d_inShape, d_outShape, d_starts, d_steps;
    for (int i = 0; i < rank; ++i) {
        d_inShape.data[i] = inShape[i];
        d_outShape.data[i] = outShape[i];
        d_starts.data[i] = starts[i];
        d_steps.data[i] = steps[i];
        totalElements *= outShape[i];
    }
    if (totalElements == 0) return;
    const int blockSize = 256;
    const int numBlocks = (int)((totalElements + blockSize - 1) / blockSize);
    slice_kernel << <numBlocks, blockSize >> > (input, output, d_inShape, d_outShape, d_starts, d_steps, rank, totalElements);
}

EXPORT void NativeSliceGrad(const float* gradOut, float* gradIn,
    const int* originalShape, const int* newShape,
    const int* starts, const int* steps, int rank) {
    if (rank <= 0 || rank > 12) return;
    long long totalElements = 1;
    SliceInfo d_origShape, d_newShape, d_starts, d_steps;
    for (int i = 0; i < rank; ++i) {
        d_origShape.data[i] = originalShape[i];
        d_newShape.data[i] = newShape[i];
        d_starts.data[i] = starts[i];
        d_steps.data[i] = steps[i];
        totalElements *= newShape[i];
    }
    if (totalElements == 0) return;
    const int blockSize = 256;
    const int numBlocks = (int)((totalElements + blockSize - 1) / blockSize);
    slice_grad_kernel << <numBlocks, blockSize >> > (gradOut, gradIn, d_origShape, d_newShape, d_starts, d_steps, rank, totalElements);
}

EXPORT void NativeConcat(const float** inputs, float* output, int numInputs,
    int outerSize, const int* concatSizes, int innerSize) {
    int totalConcatSize = 0;
    for (int k = 0; k < numInputs; k++) {
        totalConcatSize += concatSizes[k];
    }

    int currentOffset = 0;
    for (int k = 0; k < numInputs; k++) {
        int inputConcatSize = concatSizes[k];
        int totalElements = outerSize * inputConcatSize * innerSize;
        if (totalElements > 0) {
            int bs = 256;
            int gs = (totalElements + bs - 1) / bs;
            concat_copy_kernel << <gs, bs >> > (inputs[k], output, outerSize, inputConcatSize, totalConcatSize, innerSize, currentOffset);
        }
        currentOffset += inputConcatSize;
    }
}

EXPORT int InvokeHolonomicKernel(const ComplexDouble* inputs, const ComplexDouble* weights,
    const ComplexDouble* intWeights, ComplexDouble* outputs,
    int inputSize, int neuronCount, int fractalDepth) {
    int bs = 256;
    int gs = (neuronCount + bs - 1) / bs;
    holonomic_kernel << <gs, bs >> > (inputs, weights, intWeights, outputs, inputSize, neuronCount, fractalDepth);
    return 0;
}

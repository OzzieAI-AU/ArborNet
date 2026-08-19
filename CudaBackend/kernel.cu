#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define TILE_DIM 16

static thread_local cudaError_t g_lastError = cudaSuccess;

static inline void record_launch()
{
	g_lastError = cudaGetLastError();
}

static inline void get_launch_config(long long n, int* blockSize, int* gridSize)
{
	*blockSize = 256;
	if (n <= 0)
	{
		*gridSize = 0;
		return;
	}
	long long gs = (n + 255) / 256;
	if (gs > 2147483647LL) gs = 2147483647LL;
	*gridSize = (int)gs;
}

struct ShapeInfo { int data[12]; };
struct SliceInfo { int data[12]; };

struct ComplexDouble
{
	double real;
	double imag;
};

__device__ inline ComplexDouble complex_add(ComplexDouble a, ComplexDouble b)
{
	return { a.real + b.real, a.imag + b.imag };
}

__device__ inline ComplexDouble complex_mul(ComplexDouble a, ComplexDouble b)
{
	return { a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real };
}

// tanh(x+iy) = (sinh(2x) + i sin(2y)) / (cosh(2x) + cos(2y))
__device__ inline ComplexDouble complex_tanh(ComplexDouble z)
{
	double x2 = 2.0 * z.real;
	double y2 = 2.0 * z.imag;
	double den = cosh(x2) + cos(y2);
	if (den == 0.0 || isnan(den) || isinf(den))
		return { 0.0, 0.0 };
	return { sinh(x2) / den, sin(y2) / den };
}

__device__ inline uint32_t xorshift32(uint32_t* state)
{
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

// =================================================================================
// 1. ELEMENTWISE
// =================================================================================

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = a[i] + b[i];
}

__global__ void sub_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = a[i] - b[i];
}

__global__ void mul_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = a[i] * b[i];
}

__global__ void div_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = a[i] / b[i]; // IEEE inf/nan
}

__global__ void pow_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = powf(a[i], b[i]);
}

__global__ void equal_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = (fabsf(a[i] - b[i]) < 1e-6f) ? 1.0f : 0.0f;
}

__global__ void greater_than_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}

__global__ void greater_equal_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
}

__global__ void less_than_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}

__global__ void less_equal_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
}

__global__ void where_kernel(const float* __restrict__ cond, const float* __restrict__ a,
	const float* __restrict__ b, float* __restrict__ c, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		c[i] = (cond[i] != 0.0f) ? a[i] : b[i];
}

__global__ void relu_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = fmaxf(0.0f, in[i]);
}

__global__ void sigmoid_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		float x = in[i];
		out[i] = (x >= 0.0f)
			? 1.0f / (1.0f + expf(-x))
			: expf(x) / (1.0f + expf(x));
	}
}

__global__ void tanh_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = tanhf(in[i]);
}

__global__ void negate_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = -in[i];
}

__global__ void exp_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = expf(in[i]);
}

__global__ void log_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = logf(in[i]);
}

__global__ void sqrt_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = sqrtf(in[i]);
}

__global__ void abs_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = fabsf(in[i]);
}

__global__ void sin_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = sinf(in[i]);
}

__global__ void cos_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = cosf(in[i]);
}

__global__ void sign_kernel(const float* __restrict__ input, float* __restrict__ output, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		float val = input[i];
		output[i] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
	}
}

__global__ void set_scalar_kernel(float* __restrict__ data, float value, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		data[i] = value;
}

__global__ void pow_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n, float exponent)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = powf(in[i], exponent);
}

__global__ void mul_scalar_kernel(float* __restrict__ out, float scalar, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] *= scalar;
}

__global__ void add_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n, float val)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = in[i] + val;
}

__global__ void add_scalar_inplace_kernel(float* __restrict__ data, float val, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		data[i] += val;
}

__global__ void sub_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n, float val)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = in[i] - val;
}

__global__ void sub_scalar_inplace_kernel(float* __restrict__ data, float val, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		data[i] -= val;
}

__global__ void mul_scalar_out_kernel(const float* __restrict__ in, float* __restrict__ out, int n, float val)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = in[i] * val;
}

__global__ void mul_scalar_inplace_kernel(float* __restrict__ data, float val, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		data[i] *= val;
}

__global__ void div_scalar_kernel(const float* __restrict__ in, float* __restrict__ out, int n, float val)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		out[i] = in[i] / val;
}

__global__ void set_eye_kernel(float* __restrict__ data, int size)
{
	int total = size * size;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int r = i / size;
		int c = i % size;
		data[i] = (r == c) ? 1.0f : 0.0f;
	}
}

__global__ void rand_kernel(float* __restrict__ data, int n, uint32_t seed)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		uint32_t state = seed ^ (uint32_t)(i * 747796405u + 2891336453u);
		if (state == 0) state = 0xA341316Cu;
		uint32_t r = xorshift32(&state);
		data[i] = (r >> 8) * (1.0f / 16777216.0f);
	}
}

__global__ void randn_kernel(float* __restrict__ data, int n, uint32_t seed)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		uint32_t state = seed ^ (uint32_t)(i * 747796405u + 2891336453u);
		if (state == 0) state = 0xA341316Cu;
		float u1 = ((xorshift32(&state) >> 8) + 1) * (1.0f / 16777217.0f);
		float u2 = (xorshift32(&state) >> 8) * (1.0f / 16777216.0f);
		data[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
	}
}

// =================================================================================
// 2. STRUCTURE / REDUCE
// =================================================================================

__global__ void transpose_2d_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols)
{
	int total = rows * cols;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int r = i / cols;
		int c = i % cols;
		out[c * rows + r] = in[i];
	}
}

__global__ void GeneralTransposeKernel(const float* __restrict__ input, float* __restrict__ output,
	ShapeInfo shape, ShapeInfo outShape, ShapeInfo perm, int rank, long long totalElements)
{
	for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
		idx < totalElements;
		idx += (long long)blockDim.x * gridDim.x)
	{
		int outCoords[12];
		long long remainder = idx;
		for (int i = rank - 1; i >= 0; --i)
		{
			outCoords[i] = (int)(remainder % outShape.data[i]);
			remainder /= outShape.data[i];
		}

		int inCoords[12];
		for (int i = 0; i < rank; ++i)
			inCoords[perm.data[i]] = outCoords[i];

		long long inputIdx = 0;
		long long stride = 1;
		for (int i = rank - 1; i >= 0; --i)
		{
			inputIdx += (long long)inCoords[i] * stride;
			stride *= shape.data[i];
		}
		output[idx] = input[inputIdx];
	}
}

__global__ void BroadcastKernel(const float* __restrict__ input, float* __restrict__ output,
	ShapeInfo inShape, ShapeInfo outShape, int rank, long long totalElements)
{
	for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
		idx < totalElements;
		idx += (long long)blockDim.x * gridDim.x)
	{
		int outCoords[12];
		long long remainder = idx;
		for (int i = rank - 1; i >= 0; --i)
		{
			outCoords[i] = (int)(remainder % outShape.data[i]);
			remainder /= outShape.data[i];
		}

		long long inIdx = 0;
		long long inStride = 1;
		for (int i = rank - 1; i >= 0; --i)
		{
			int inCoord = (inShape.data[i] == 1) ? 0 : outCoords[i];
			inIdx += inCoord * inStride;
			inStride *= inShape.data[i];
		}
		output[idx] = input[inIdx];
	}
}

__global__ void sum_to_kernel(const float* __restrict__ input, float* __restrict__ output,
	ShapeInfo inShape, ShapeInfo outShape, int rank, long long n)
{
	for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
		idx < n;
		idx += (long long)blockDim.x * gridDim.x)
	{
		int coords[12];
		long long rem = idx;
		for (int i = rank - 1; i >= 0; --i)
		{
			coords[i] = (int)(rem % inShape.data[i]);
			rem /= inShape.data[i];
		}

		long long outIdx = 0;
		long long stride = 1;
		for (int i = rank - 1; i >= 0; --i)
		{
			int c = (outShape.data[i] == 1) ? 0 : coords[i];
			outIdx += c * stride;
			stride *= outShape.data[i];
		}
		atomicAdd(&output[outIdx], input[idx]);
	}
}

__global__ void slice_kernel(const float* __restrict__ input, float* __restrict__ output,
	SliceInfo inShape, SliceInfo outShape, SliceInfo starts, SliceInfo steps,
	int rank, long long totalElements)
{
	for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
		idx < totalElements;
		idx += (long long)blockDim.x * gridDim.x)
	{
		int outCoords[12];
		long long remainder = idx;
		for (int i = rank - 1; i >= 0; --i)
		{
			outCoords[i] = (int)(remainder % outShape.data[i]);
			remainder /= outShape.data[i];
		}

		long long inIdx = 0;
		long long stride = 1;
		for (int i = rank - 1; i >= 0; --i)
		{
			int inCoord = starts.data[i] + outCoords[i] * steps.data[i];
			inIdx += (long long)inCoord * stride;
			stride *= inShape.data[i];
		}
		output[idx] = input[inIdx];
	}
}

__global__ void slice_grad_kernel(const float* __restrict__ gradOut, float* __restrict__ gradIn,
	SliceInfo originalShape, SliceInfo newShape, SliceInfo starts, SliceInfo steps,
	int rank, long long totalElements)
{
	for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
		idx < totalElements;
		idx += (long long)blockDim.x * gridDim.x)
	{
		int outCoords[12];
		long long remainder = idx;
		for (int i = rank - 1; i >= 0; --i)
		{
			outCoords[i] = (int)(remainder % newShape.data[i]);
			remainder /= newShape.data[i];
		}

		long long inIdx = 0;
		long long stride = 1;
		for (int i = rank - 1; i >= 0; --i)
		{
			int inCoord = starts.data[i] + outCoords[i] * steps.data[i];
			inIdx += (long long)inCoord * stride;
			stride *= originalShape.data[i];
		}
		atomicAdd(&gradIn[inIdx], gradOut[idx]);
	}
}

__global__ void concat_copy_kernel(const float* __restrict__ input, float* __restrict__ output,
	int outerSize, int inputConcatSize, int totalConcatSize, int innerSize, int currentOffset)
{
	int totalElements = outerSize * inputConcatSize * innerSize;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElements; i += blockDim.x * gridDim.x)
	{
		int o = i / (inputConcatSize * innerSize);
		int rem = i % (inputConcatSize * innerSize);
		int c = rem / innerSize;
		int in_val = rem % innerSize;
		int outIdx = o * (totalConcatSize * innerSize) + (currentOffset + c) * innerSize + in_val;
		output[outIdx] = input[i];
	}
}

__global__ void gather_axis_kernel(const float* __restrict__ input, const float* __restrict__ indices,
	float* __restrict__ output, int outer, int dim, int inner, int k)
{
	int total = outer * k * inner;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
	{
		int o = idx / (k * inner);
		int rem = idx % (k * inner);
		int ki = rem / inner;
		int i = rem % inner;
		int classIdx = (int)indices[o * k * inner + ki * inner + i];
		if (classIdx >= 0 && classIdx < dim)
			output[idx] = input[(o * dim + classIdx) * inner + i];
		else
			output[idx] = 0.0f;
	}
}

__global__ void gather_axis_grad_kernel(const float* __restrict__ gradOut, const float* __restrict__ indices,
	float* __restrict__ gradIn, int outer, int dim, int inner, int k)
{
	int total = outer * k * inner;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
	{
		int o = idx / (k * inner);
		int rem = idx % (k * inner);
		int ki = rem / inner;
		int i = rem % inner;
		int classIdx = (int)indices[o * k * inner + ki * inner + i];
		if (classIdx >= 0 && classIdx < dim)
			atomicAdd(&gradIn[(o * dim + classIdx) * inner + i], gradOut[idx]);
	}
}

__global__ void embedding_forward_kernel(const float* __restrict__ weights, const float* __restrict__ indices,
	float* __restrict__ output, int num_words, int embed_dim, int total_indices)
{
	int total = total_indices * embed_dim;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int word_idx = i / embed_dim;
		int dim_idx = i % embed_dim;
		int token = (int)indices[word_idx];
		output[i] = (token >= 0 && token < num_words) ? weights[token * embed_dim + dim_idx] : 0.0f;
	}
}

__global__ void embedding_backward_kernel(const float* __restrict__ grad_out, const float* __restrict__ indices,
	float* __restrict__ grad_weights, int num_words, int embed_dim, int total_indices)
{
	int total = total_indices * embed_dim;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int word_idx = i / embed_dim;
		int dim_idx = i % embed_dim;
		int token = (int)indices[word_idx];
		if (token >= 0 && token < num_words)
			atomicAdd(&grad_weights[token * embed_dim + dim_idx], grad_out[i]);
	}
}

__global__ void sum_all_reduction_kernel(const float* __restrict__ in, float* __restrict__ out, int n)
{
	__shared__ float sdata[256];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	float local_sum = 0.0f;
	while (i < (unsigned int)n)
	{
		local_sum += in[i];
		i += gridDim.x * blockDim.x;
	}
	sdata[tid] = local_sum;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void mean_axis_kernel(const float* __restrict__ input, float* __restrict__ output, int outer, int dim, int inner)
{
	int total = outer * inner;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int o = i / inner;
		int in_val = i % inner;
		float sum = 0.0f;
		for (int d = 0; d < dim; d++)
			sum += input[(o * dim + d) * inner + in_val];
		output[i] = sum / (float)dim;
	}
}

__global__ void sum_axis_kernel(const float* __restrict__ input, float* __restrict__ output, int outer, int dim, int inner)
{
	int total = outer * inner;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int o = i / inner;
		int in_val = i % inner;
		float sum = 0.0f;
		for (int d = 0; d < dim; d++)
			sum += input[(o * dim + d) * inner + in_val];
		output[i] = sum;
	}
}

__global__ void argmax_kernel(const float* __restrict__ input, float* __restrict__ output, int outer, int dim, int inner)
{
	int total = outer * inner;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int o = i / inner;
		int in_val = i % inner;
		float max_val = -FLT_MAX;
		int max_idx = 0;
		for (int d = 0; d < dim; d++)
		{
			float val = input[(o * dim + d) * inner + in_val];
			if (val > max_val) { max_val = val; max_idx = d; }
		}
		output[i] = (float)max_idx;
	}
}

__global__ void argmin_kernel(const float* __restrict__ input, float* __restrict__ output, int outer, int dim, int inner)
{
	int total = outer * inner;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int o = i / inner;
		int in_val = i % inner;
		float min_val = FLT_MAX;
		int min_idx = 0;
		for (int d = 0; d < dim; d++)
		{
			float val = input[(o * dim + d) * inner + in_val];
			if (val < min_val) { min_val = val; min_idx = d; }
		}
		output[i] = (float)min_idx;
	}
}

__global__ void max_axis_kernel(const float* __restrict__ input, float* __restrict__ output, int outer, int dim, int inner)
{
	int total = outer * inner;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int o = i / inner;
		int in_val = i % inner;
		float max_val = -FLT_MAX;
		for (int d = 0; d < dim; d++)
		{
			float val = input[(o * dim + d) * inner + in_val];
			if (val > max_val) max_val = val;
		}
		output[i] = max_val;
	}
}

__global__ void min_axis_kernel(const float* __restrict__ input, float* __restrict__ output, int outer, int dim, int inner)
{
	int total = outer * inner;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int o = i / inner;
		int in_val = i % inner;
		float min_val = FLT_MAX;
		for (int d = 0; d < dim; d++)
		{
			float val = input[(o * dim + d) * inner + in_val];
			if (val < min_val) min_val = val;
		}
		output[i] = min_val;
	}
}

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int outer, int dim, int inner)
{
	int total = outer * inner;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int o = i / inner;
		int in_val = i % inner;
		double sum = 0.0;
		for (int d = 0; d < dim; d++)
		{
			int pos = (o * dim + d) * inner + in_val;
			sum += (double)input[pos];
			output[pos] = (float)sum;
		}
	}
}

__global__ void reverse_cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int outer, int dim, int inner)
{
	int total = outer * inner;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x)
	{
		int o = i / inner;
		int in_val = i % inner;
		double sum = 0.0;
		for (int d = dim - 1; d >= 0; --d)
		{
			int pos = (o * dim + d) * inner + in_val;
			sum += (double)input[pos];
			output[pos] = (float)sum;
		}
	}
}

__global__ void matmul_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
	float* __restrict__ C, int m, int n, int k)
{
	__shared__ float s_A[TILE_DIM][TILE_DIM + 1];
	__shared__ float s_B[TILE_DIM][TILE_DIM + 1];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	int row = by * TILE_DIM + ty;
	int col = bx * TILE_DIM + tx;
	float sum = 0.0f;

	for (int q = 0; q < (k + TILE_DIM - 1) / TILE_DIM; ++q)
	{
		s_A[ty][tx] = (row < m && (q * TILE_DIM + tx) < k) ? A[row * k + q * TILE_DIM + tx] : 0.0f;
		s_B[ty][tx] = (col < n && (q * TILE_DIM + ty) < k) ? B[(q * TILE_DIM + ty) * n + col] : 0.0f;
		__syncthreads();
#pragma unroll
		for (int j = 0; j < TILE_DIM; ++j)
			sum += s_A[ty][j] * s_B[j][tx];
		__syncthreads();
	}
	if (row < m && col < n)
		C[row * n + col] = sum;
}

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols)
{
	for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x)
	{
		const float* inRow = input + r * cols;
		float* outRow = output + r * cols;
		float maxVal = -FLT_MAX;
		for (int i = 0; i < cols; i++)
			if (inRow[i] > maxVal) maxVal = inRow[i];

		float sum = 0.0f;
		for (int i = 0; i < cols; i++)
		{
			float e = expf(inRow[i] - maxVal);
			outRow[i] = e;
			sum += e;
		}
		float inv = (sum == 0.0f) ? 0.0f : 1.0f / sum;
		for (int i = 0; i < cols; i++)
			outRow[i] *= inv;
	}
}

__global__ void relu_grad_kernel(const float* __restrict__ gradOut, const float* __restrict__ originIn,
	float* __restrict__ gradIn, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		gradIn[i] = (originIn[i] > 0.0f) ? gradOut[i] : 0.0f;
}

__global__ void sigmoid_grad_kernel(const float* __restrict__ gradOut, const float* __restrict__ originOut,
	float* __restrict__ gradIn, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		float s = originOut[i];
		gradIn[i] = gradOut[i] * s * (1.0f - s);
	}
}

__global__ void logical_not_kernel(const float* __restrict__ input, float* __restrict__ output, int n)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		output[i] = (input[i] == 0.0f) ? 1.0f : 0.0f;
}

__global__ void clip_kernel(const float* __restrict__ input, float* __restrict__ output, int n, float minVal, float maxVal)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		output[i] = fminf(fmaxf(input[i], minVal), maxVal);
}

__global__ void clip_mask_kernel(const float* __restrict__ input, float* __restrict__ output, int n, float minVal, float maxVal)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
		output[i] = (input[i] >= minVal && input[i] <= maxVal) ? 1.0f : 0.0f;
}

__global__ void holonomic_kernel(const ComplexDouble* __restrict__ inputs, const ComplexDouble* __restrict__ weights,
	const ComplexDouble* __restrict__ intWeights, ComplexDouble* __restrict__ outputs,
	int inputSize, int neuronCount, int fractalDepth)
{
	for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < neuronCount; n += blockDim.x * gridDim.x)
	{
		ComplexDouble psi = { 0.0, 0.0 };
		for (int i = 0; i < inputSize; i++)
			psi = complex_add(psi, complex_mul(inputs[i], weights[n * inputSize + i]));

		ComplexDouble z = { 0.0, 0.0 };
		for (int t = 0; t < fractalDepth; t++)
			z = complex_tanh(complex_add(complex_mul(intWeights[n], z), psi));
		outputs[n] = z;
	}
}

// =================================================================================
// 3. CONV
// =================================================================================

__global__ void conv2d_forward_kernel(
	const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output,
	int batch, int in_channels, int in_h, int in_w,
	int out_channels, int out_h, int out_w,
	int k_h, int k_w, int stride, int padding)
{
	int total_elements = batch * out_channels * out_h * out_w;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x)
	{
		int w_out = idx % out_w;
		int h_out = (idx / out_w) % out_h;
		int c_out = (idx / (out_w * out_h)) % out_channels;
		int b = idx / (out_w * out_h * out_channels);

		float sum = 0.0f;
		for (int c_in = 0; c_in < in_channels; ++c_in)
		{
			for (int kh = 0; kh < k_h; ++kh)
			{
				int h_in = h_out * stride - padding + kh;
				if ((unsigned)h_in >= (unsigned)in_h) continue;
				for (int kw = 0; kw < k_w; ++kw)
				{
					int w_in = w_out * stride - padding + kw;
					if ((unsigned)w_in >= (unsigned)in_w) continue;
					int input_offset = ((b * in_channels + c_in) * in_h + h_in) * in_w + w_in;
					int weight_offset = ((c_out * in_channels + c_in) * k_h + kh) * k_w + kw;
					sum += input[input_offset] * weight[weight_offset];
				}
			}
		}
		output[idx] = sum;
	}
}

__global__ void conv2d_grad_weight_kernel(
	const float* __restrict__ input, const float* __restrict__ grad_out, float* __restrict__ grad_weight,
	int batch, int in_channels, int in_h, int in_w,
	int out_channels, int out_h, int out_w,
	int k_h, int k_w, int stride, int padding)
{
	int total_weights = out_channels * in_channels * k_h * k_w;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_weights; idx += blockDim.x * gridDim.x)
	{
		int kw = idx % k_w;
		int kh = (idx / k_w) % k_h;
		int c_in = (idx / (k_w * k_h)) % in_channels;
		int c_out = idx / (k_w * k_h * in_channels);

		float sum = 0.0f;
		for (int b = 0; b < batch; ++b)
		{
			for (int oh = 0; oh < out_h; ++oh)
			{
				int ih = oh * stride - padding + kh;
				if ((unsigned)ih >= (unsigned)in_h) continue;
				for (int ow = 0; ow < out_w; ++ow)
				{
					int iw = ow * stride - padding + kw;
					if ((unsigned)iw >= (unsigned)in_w) continue;
					int input_offset = ((b * in_channels + c_in) * in_h + ih) * in_w + iw;
					int grad_offset = ((b * out_channels + c_out) * out_h + oh) * out_w + ow;
					sum += input[input_offset] * grad_out[grad_offset];
				}
			}
		}
		grad_weight[idx] = sum;
	}
}

__global__ void conv2d_grad_input_kernel(
	const float* __restrict__ grad_out, const float* __restrict__ weight, float* __restrict__ grad_input,
	int batch, int in_channels, int in_h, int in_w,
	int out_channels, int out_h, int out_w,
	int k_h, int k_w, int stride, int padding)
{
	int total_elements = batch * in_channels * in_h * in_w;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x)
	{
		int w_in = idx % in_w;
		int h_in = (idx / in_w) % in_h;
		int c_in = (idx / (in_w * in_h)) % in_channels;
		int b = idx / (in_w * in_h * in_channels);

		float sum = 0.0f;
		for (int c_out = 0; c_out < out_channels; ++c_out)
		{
			for (int kh = 0; kh < k_h; ++kh)
			{
				int h_tmp = h_in + padding - kh;
				if (h_tmp % stride != 0) continue;
				int h_out = h_tmp / stride;
				if ((unsigned)h_out >= (unsigned)out_h) continue;
				for (int kw = 0; kw < k_w; ++kw)
				{
					int w_tmp = w_in + padding - kw;
					if (w_tmp % stride != 0) continue;
					int w_out = w_tmp / stride;
					if ((unsigned)w_out >= (unsigned)out_w) continue;
					int grad_offset = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
					int weight_offset = ((c_out * in_channels + c_in) * k_h + kh) * k_w + kw;
					sum += grad_out[grad_offset] * weight[weight_offset];
				}
			}
		}
		grad_input[idx] = sum;
	}
}

__global__ void conv3d_forward_kernel(
	const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output,
	int batch, int in_channels, int in_d, int in_h, int in_w,
	int out_channels, int out_d, int out_h, int out_w,
	int k_d, int k_h, int k_w, int stride, int padding)
{
	int total_elements = batch * out_channels * out_d * out_h * out_w;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x)
	{
		int w_out = idx % out_w;
		int h_out = (idx / out_w) % out_h;
		int d_out = (idx / (out_w * out_h)) % out_d;
		int c_out = (idx / (out_w * out_h * out_d)) % out_channels;
		int b = idx / (out_w * out_h * out_d * out_channels);

		float sum = 0.0f;
		for (int c_in = 0; c_in < in_channels; ++c_in)
		{
			for (int kd = 0; kd < k_d; ++kd)
			{
				int d_in = d_out * stride - padding + kd;
				if ((unsigned)d_in >= (unsigned)in_d) continue;
				for (int kh = 0; kh < k_h; ++kh)
				{
					int h_in = h_out * stride - padding + kh;
					if ((unsigned)h_in >= (unsigned)in_h) continue;
					for (int kw = 0; kw < k_w; ++kw)
					{
						int w_in = w_out * stride - padding + kw;
						if ((unsigned)w_in >= (unsigned)in_w) continue;
						int input_offset = (((b * in_channels + c_in) * in_d + d_in) * in_h + h_in) * in_w + w_in;
						int weight_offset = (((c_out * in_channels + c_in) * k_d + kd) * k_h + kh) * k_w + kw;
						sum += input[input_offset] * weight[weight_offset];
					}
				}
			}
		}
		output[idx] = sum;
	}
}

__global__ void conv3d_grad_weight_kernel(
	const float* __restrict__ input, const float* __restrict__ grad_out, float* __restrict__ grad_weight,
	int batch, int in_channels, int in_d, int in_h, int in_w,
	int out_channels, int out_d, int out_h, int out_w,
	int k_d, int k_h, int k_w, int stride, int padding)
{
	int total_weights = out_channels * in_channels * k_d * k_h * k_w;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_weights; idx += blockDim.x * gridDim.x)
	{
		int kw = idx % k_w;
		int kh = (idx / k_w) % k_h;
		int kd = (idx / (k_w * k_h)) % k_d;
		int c_in = (idx / (k_w * k_h * k_d)) % in_channels;
		int c_out = idx / (k_w * k_h * k_d * in_channels);

		float sum = 0.0f;
		for (int b = 0; b < batch; ++b)
		{
			for (int od = 0; od < out_d; ++od)
			{
				int id = od * stride - padding + kd;
				if ((unsigned)id >= (unsigned)in_d) continue;
				for (int oh = 0; oh < out_h; ++oh)
				{
					int ih = oh * stride - padding + kh;
					if ((unsigned)ih >= (unsigned)in_h) continue;
					for (int ow = 0; ow < out_w; ++ow)
					{
						int iw = ow * stride - padding + kw;
						if ((unsigned)iw >= (unsigned)in_w) continue;
						int input_offset = (((b * in_channels + c_in) * in_d + id) * in_h + ih) * in_w + iw;
						int grad_offset = (((b * out_channels + c_out) * out_d + od) * out_h + oh) * out_w + ow;
						sum += input[input_offset] * grad_out[grad_offset];
					}
				}
			}
		}
		grad_weight[idx] = sum;
	}
}

__global__ void conv3d_grad_input_kernel(
	const float* __restrict__ grad_out, const float* __restrict__ weight, float* __restrict__ grad_input,
	int batch, int in_channels, int in_d, int in_h, int in_w,
	int out_channels, int out_d, int out_h, int out_w,
	int k_d, int k_h, int k_w, int stride, int padding)
{
	int total_elements = batch * in_channels * in_d * in_h * in_w;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x)
	{
		int w_in = idx % in_w;
		int h_in = (idx / in_w) % in_h;
		int d_in = (idx / (in_w * in_h)) % in_d;
		int c_in = (idx / (in_w * in_h * in_d)) % in_channels;
		int b = idx / (in_w * in_h * in_d * in_channels);

		float sum = 0.0f;
		for (int c_out = 0; c_out < out_channels; ++c_out)
		{
			for (int kd = 0; kd < k_d; ++kd)
			{
				int d_tmp = d_in + padding - kd;
				if (d_tmp % stride != 0) continue;
				int d_out = d_tmp / stride;
				if ((unsigned)d_out >= (unsigned)out_d) continue;
				for (int kh = 0; kh < k_h; ++kh)
				{
					int h_tmp = h_in + padding - kh;
					if (h_tmp % stride != 0) continue;
					int h_out = h_tmp / stride;
					if ((unsigned)h_out >= (unsigned)out_h) continue;
					for (int kw = 0; kw < k_w; ++kw)
					{
						int w_tmp = w_in + padding - kw;
						if (w_tmp % stride != 0) continue;
						int w_out = w_tmp / stride;
						if ((unsigned)w_out >= (unsigned)out_w) continue;
						int grad_offset = (((b * out_channels + c_out) * out_d + d_out) * out_h + h_out) * out_w + w_out;
						int weight_offset = (((c_out * in_channels + c_in) * k_d + kd) * k_h + kh) * k_w + kw;
						sum += grad_out[grad_offset] * weight[weight_offset];
					}
				}
			}
		}
		grad_input[idx] = sum;
	}
}

__global__ void topk_kernel(
	const float* __restrict__ input, float* __restrict__ outValues, float* __restrict__ outIndices,
	int outer, int dim, int inner, int k)
{
	int total = outer * inner;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
	{
		int o = idx / inner;
		int i = idx % inner;
		const float* slice = input + (o * dim * inner + i);

		for (int r = 0; r < k; ++r)
		{
			float bestVal = -FLT_MAX;
			int bestIdx = 0;
			for (int d = 0; d < dim; ++d)
			{
				bool taken = false;
				for (int prev = 0; prev < r; ++prev)
				{
					int prevOutIdx = o * k * inner + prev * inner + i;
					if ((int)outIndices[prevOutIdx] == d) { taken = true; break; }
				}
				if (taken) continue;
				float val = slice[d * inner];
				if (val > bestVal) { bestVal = val; bestIdx = d; }
			}
			int outIdx = o * k * inner + r * inner + i;
			outValues[outIdx] = bestVal;
			outIndices[outIdx] = (float)bestIdx;
		}
	}
}

__global__ void topk_scatter_grad_kernel(
	const float* __restrict__ gradOut, const float* __restrict__ indices, float* __restrict__ gradIn,
	int outer, int dim, int inner, int k)
{
	int total = outer * inner * k;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x)
	{
		int o = idx / (k * inner);
		int rem = idx % (k * inner);
		int r = rem / inner;
		int i = rem % inner;
		int origIdx = (int)indices[o * k * inner + r * inner + i];
		if (origIdx >= 0 && origIdx < dim)
			atomicAdd(&gradIn[o * dim * inner + origIdx * inner + i], gradOut[o * k * inner + r * inner + i]);
	}
}

// =================================================================================
// 4. HOST DISPATCH
// =================================================================================

#define LAUNCH1(n, kernel, ...) \
    do { \
        if ((n) <= 0) { g_lastError = cudaSuccess; return; } \
        int bs, gs; get_launch_config((n), &bs, &gs); \
        kernel<<<gs, bs>>>(__VA_ARGS__); \
        record_launch(); \
    } while (0)

EXPORT int NativeGetLastError(void) { return (int)g_lastError; }

EXPORT void NativeAdd(const float* a, const float* b, float* c, int n) { LAUNCH1(n, add_kernel, a, b, c, n); }
EXPORT void NativeSubtract(const float* a, const float* b, float* c, int n) { LAUNCH1(n, sub_kernel, a, b, c, n); }
EXPORT void NativeMultiply(const float* a, const float* b, float* c, int n) { LAUNCH1(n, mul_kernel, a, b, c, n); }
EXPORT void NativeDivide(const float* a, const float* b, float* c, int n) { LAUNCH1(n, div_kernel, a, b, c, n); }
EXPORT void NativePow(const float* a, const float* b, float* c, int n) { LAUNCH1(n, pow_kernel, a, b, c, n); }
EXPORT void NativePowTensor(const float* a, const float* b, float* c, int n) { LAUNCH1(n, pow_kernel, a, b, c, n); }
EXPORT void NativeEqual(const float* a, const float* b, float* c, int n) { LAUNCH1(n, equal_kernel, a, b, c, n); }
EXPORT void NativeGreaterThan(const float* a, const float* b, float* c, int n) { LAUNCH1(n, greater_than_kernel, a, b, c, n); }
EXPORT void NativeGreaterThanOrEqual(const float* a, const float* b, float* c, int n) { LAUNCH1(n, greater_equal_kernel, a, b, c, n); }
EXPORT void NativeLessThan(const float* a, const float* b, float* c, int n) { LAUNCH1(n, less_than_kernel, a, b, c, n); }
EXPORT void NativeLessEqual(const float* a, const float* b, float* c, int n) { LAUNCH1(n, less_equal_kernel, a, b, c, n); }
EXPORT void NativeWhere(const float* cond, const float* a, const float* b, float* c, int n) { LAUNCH1(n, where_kernel, cond, a, b, c, n); }
EXPORT void NativeReLU(const float* in, float* out, int n) { LAUNCH1(n, relu_kernel, in, out, n); }
EXPORT void NativeReLUGrad(const float* gradOut, const float* originIn, float* gradIn, int n) { LAUNCH1(n, relu_grad_kernel, gradOut, originIn, gradIn, n); }
EXPORT void NativeSigmoid(const float* in, float* out, int n) { LAUNCH1(n, sigmoid_kernel, in, out, n); }
EXPORT void NativeSigmoidGrad(const float* gradOut, const float* originOut, float* gradIn, int n) { LAUNCH1(n, sigmoid_grad_kernel, gradOut, originOut, gradIn, n); }
EXPORT void NativeTanh(const float* in, float* out, int n) { LAUNCH1(n, tanh_kernel, in, out, n); }
EXPORT void NativeNegate(const float* in, float* out, int n) { LAUNCH1(n, negate_kernel, in, out, n); }
EXPORT void NativeExp(const float* in, float* out, int n) { LAUNCH1(n, exp_kernel, in, out, n); }
EXPORT void NativeLog(const float* in, float* out, int n) { LAUNCH1(n, log_kernel, in, out, n); }
EXPORT void NativeSqrt(const float* in, float* out, int n) { LAUNCH1(n, sqrt_kernel, in, out, n); }
EXPORT void NativeAbs(const float* in, float* out, int n) { LAUNCH1(n, abs_kernel, in, out, n); }
EXPORT void NativeSin(const float* in, float* out, int n) { LAUNCH1(n, sin_kernel, in, out, n); }
EXPORT void NativeCos(const float* in, float* out, int n) { LAUNCH1(n, cos_kernel, in, out, n); }
EXPORT void NativeSign(const float* input, float* output, int n) { LAUNCH1(n, sign_kernel, input, output, n); }
EXPORT void NativeOnes(float* out, int n) { LAUNCH1(n, set_scalar_kernel, out, 1.0f, n); }
EXPORT void NativeSetScalar(float* out, float value, int n) { LAUNCH1(n, set_scalar_kernel, out, value, n); }
EXPORT void NativePowScalar(const float* in, float* out, int n, float exponent) { LAUNCH1(n, pow_scalar_kernel, in, out, n, exponent); }
EXPORT void NativeAddScalar(const float* in, float* out, int n, float value) { LAUNCH1(n, add_scalar_kernel, in, out, n, value); }
EXPORT void NativeSubtractScalar(const float* in, float* out, int n, float value) { LAUNCH1(n, sub_scalar_kernel, in, out, n, value); }
EXPORT void NativeMultiplyScalar(const float* in, float* out, int n, float value) { LAUNCH1(n, mul_scalar_out_kernel, in, out, n, value); }
EXPORT void NativeDivideScalar(const float* in, float* out, int n, float value) { LAUNCH1(n, div_scalar_kernel, in, out, n, value); }
EXPORT void NativeAddScalarInPlace(float* data, float value, int n) { LAUNCH1(n, add_scalar_inplace_kernel, data, value, n); }
EXPORT void NativeSubtractScalarInPlace(float* data, float value, int n) { LAUNCH1(n, sub_scalar_inplace_kernel, data, value, n); }
EXPORT void NativeMultiplyScalarInPlace(float* data, float value, int n) { LAUNCH1(n, mul_scalar_inplace_kernel, data, value, n); }

EXPORT void NativeEye(float* data, int size)
{
	if (size <= 0) { g_lastError = cudaSuccess; return; }
	long long total = (long long)size * size;
	LAUNCH1(total, set_eye_kernel, data, size);
}

EXPORT void NativeRand(float* data, int n, unsigned int seed) { LAUNCH1(n, rand_kernel, data, n, seed); }
EXPORT void NativeRandn(float* data, int n, unsigned int seed) { LAUNCH1(n, randn_kernel, data, n, seed); }
EXPORT void NativeTranspose(const float* in, float* out, int rows, int cols) { LAUNCH1((long long)rows * cols, transpose_2d_kernel, in, out, rows, cols); }

EXPORT void NativeSumAll(const float* in, float* out, int n)
{
	if (n <= 0) { g_lastError = cudaSuccess; return; }
	int bs = 256;
	int gs = (int)(((long long)n + bs - 1) / bs);
	cudaError_t z = cudaMemset(out, 0, sizeof(float));
	if (z != cudaSuccess) { g_lastError = z; return; }
	sum_all_reduction_kernel << <gs, bs >> > (in, out, n);
	record_launch();
}

EXPORT void NativeMeanAll(const float* in, float* out, int n)
{
	NativeSumAll(in, out, n);
	if (g_lastError != cudaSuccess) return;
	if (n <= 0) return;
	mul_scalar_kernel << <1, 1 >> > (out, 1.0f / (float)n, 1);
	record_launch();
}

EXPORT void NativeArgMax(const float* in, float* out, int outer, int dim, int inner) { LAUNCH1((long long)outer * inner, argmax_kernel, in, out, outer, dim, inner); }
EXPORT void NativeArgMin(const float* in, float* out, int outer, int dim, int inner) { LAUNCH1((long long)outer * inner, argmin_kernel, in, out, outer, dim, inner); }
EXPORT void NativeCumSum(const float* in, float* out, int outer, int dim, int inner) { LAUNCH1((long long)outer * inner, cumsum_kernel, in, out, outer, dim, inner); }
EXPORT void NativeReverseCumSum(const float* in, float* out, int outer, int dim, int inner) { LAUNCH1((long long)outer * inner, reverse_cumsum_kernel, in, out, outer, dim, inner); }
EXPORT void NativeMeanAxis(const float* input, float* output, int outer, int dim, int inner) { LAUNCH1((long long)outer * inner, mean_axis_kernel, input, output, outer, dim, inner); }
EXPORT void NativeSumAxis(const float* input, float* output, int outer, int dim, int inner) { LAUNCH1((long long)outer * inner, sum_axis_kernel, input, output, outer, dim, inner); }
EXPORT void NativeMaxAxis(const float* input, float* output, int outer, int dim, int inner) { LAUNCH1((long long)outer * inner, max_axis_kernel, input, output, outer, dim, inner); }
EXPORT void NativeMinAxis(const float* input, float* output, int outer, int dim, int inner) { LAUNCH1((long long)outer * inner, min_axis_kernel, input, output, outer, dim, inner); }

EXPORT void NativeGeneralTranspose(const float* input, float* output, const int* shape, const int* perm, int rank)
{
	if (rank <= 0 || rank > 12) { g_lastError = cudaErrorInvalidValue; return; }
	long long totalElements = 1;
	ShapeInfo d_shape, d_outShape, d_perm;
	for (int i = 0; i < rank; ++i)
	{
		d_shape.data[i] = shape[i];
		d_outShape.data[i] = shape[perm[i]];
		d_perm.data[i] = perm[i];
		totalElements *= shape[i];
	}
	LAUNCH1(totalElements, GeneralTransposeKernel, input, output, d_shape, d_outShape, d_perm, rank, totalElements);
}

EXPORT void NativeBroadcast(const float* input, float* output, const int* inputShape, const int* outputShape, int rank)
{
	if (rank <= 0 || rank > 12) { g_lastError = cudaErrorInvalidValue; return; }
	long long totalElements = 1;
	ShapeInfo d_inShape, d_outShape;
	for (int i = 0; i < rank; ++i)
	{
		d_inShape.data[i] = inputShape[i];
		d_outShape.data[i] = outputShape[i];
		totalElements *= outputShape[i];
	}
	LAUNCH1(totalElements, BroadcastKernel, input, output, d_inShape, d_outShape, rank, totalElements);
}

EXPORT void NativeSumTo(const float* input, float* output, const int* inShape, const int* outShape, int rank)
{
	if (rank <= 0 || rank > 12 || input == nullptr || output == nullptr || inShape == nullptr || outShape == nullptr)
	{
		g_lastError = cudaErrorInvalidValue;
		return;
	}

	long long nIn = 1;
	long long nOut = 1;
	ShapeInfo d_in, d_out;
	for (int i = 0; i < rank; ++i)
	{
		if (inShape[i] <= 0 || outShape[i] <= 0)
		{
			g_lastError = cudaErrorInvalidValue;
			return;
		}
		if (outShape[i] != inShape[i] && outShape[i] != 1)
		{
			g_lastError = cudaErrorInvalidValue;
			return;
		}
		d_in.data[i] = inShape[i];
		d_out.data[i] = outShape[i];
		nIn *= inShape[i];
		nOut *= outShape[i];
	}

	cudaError_t z = cudaMemset(output, 0, (size_t)nOut * sizeof(float));
	if (z != cudaSuccess)
	{
		g_lastError = z;
		return;
	}

	LAUNCH1(nIn, sum_to_kernel, input, output, d_in, d_out, rank, nIn);
}


EXPORT void NativeMatMul(const float* A, const float* B, float* C, int m, int n, int k)
{
	if (m <= 0 || n <= 0 || k <= 0) { g_lastError = cudaSuccess; return; }
	dim3 blockSize(TILE_DIM, TILE_DIM);
	dim3 gridSize((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
	matmul_tiled_kernel << <gridSize, blockSize >> > (A, B, C, m, n, k);
	record_launch();
}

EXPORT void NativeSoftmax(const float* input, float* output, int rows, int cols)
{
	if (rows <= 0 || cols <= 0) { g_lastError = cudaSuccess; return; }
	LAUNCH1(rows, softmax_kernel, input, output, rows, cols);
}

EXPORT void NativeLogicalNot(const float* input, float* output, int count) { LAUNCH1(count, logical_not_kernel, input, output, count); }
EXPORT void NativeClip(const float* input, float* output, int count, float minVal, float maxVal) { LAUNCH1(count, clip_kernel, input, output, count, minVal, maxVal); }
EXPORT void NativeClipMask(const float* input, float* output, int count, float minVal, float maxVal) { LAUNCH1(count, clip_mask_kernel, input, output, count, minVal, maxVal); }

EXPORT void NativeGather(const float* input, const float* indices, float* output, int batch, int classes)
{
	LAUNCH1(batch, gather_axis_kernel, input, indices, output, batch, classes, 1, 1);
}

EXPORT void NativeGatherGrad(const float* gradOut, const float* indices, float* gradIn, int batch, int classes)
{
	LAUNCH1(batch, gather_axis_grad_kernel, gradOut, indices, gradIn, batch, classes, 1, 1);
}

EXPORT void NativeGatherAxis(const float* input, const float* indices, float* output, int outer, int dim, int inner, int k)
{
	LAUNCH1((long long)outer * k * inner, gather_axis_kernel, input, indices, output, outer, dim, inner, k);
}

EXPORT void NativeGatherAxisGrad(const float* gradOut, const float* indices, float* gradIn, int outer, int dim, int inner, int k)
{
	LAUNCH1((long long)outer * k * inner, gather_axis_grad_kernel, gradOut, indices, gradIn, outer, dim, inner, k);
}

EXPORT void NativeEmbedding(const float* weights, const float* indices, float* output, int num_words, int embed_dim, int total_indices)
{
	LAUNCH1((long long)total_indices * embed_dim, embedding_forward_kernel, weights, indices, output, num_words, embed_dim, total_indices);
}

EXPORT void NativeEmbeddingGrad(const float* grad_out, const float* indices, float* grad_weights, int num_words, int embed_dim, int total_indices)
{
	LAUNCH1((long long)total_indices * embed_dim, embedding_backward_kernel, grad_out, indices, grad_weights, num_words, embed_dim, total_indices);
}

EXPORT void NativeSlice(const float* input, float* output, const int* inShape, const int* outShape,
	const int* starts, const int* steps, int rank)
{
	if (rank <= 0 || rank > 12) { g_lastError = cudaErrorInvalidValue; return; }
	long long totalElements = 1;
	SliceInfo d_inShape, d_outShape, d_starts, d_steps;
	for (int i = 0; i < rank; ++i)
	{
		d_inShape.data[i] = inShape[i];
		d_outShape.data[i] = outShape[i];
		d_starts.data[i] = starts[i];
		d_steps.data[i] = steps[i];
		totalElements *= outShape[i];
	}
	LAUNCH1(totalElements, slice_kernel, input, output, d_inShape, d_outShape, d_starts, d_steps, rank, totalElements);
}

EXPORT void NativeSliceGrad(const float* gradOut, float* gradIn, const int* originalShape, const int* newShape,
	const int* starts, const int* steps, int rank)
{
	if (rank <= 0 || rank > 12) { g_lastError = cudaErrorInvalidValue; return; }
	long long totalElements = 1;
	SliceInfo d_origShape, d_newShape, d_starts, d_steps;
	for (int i = 0; i < rank; ++i)
	{
		d_origShape.data[i] = originalShape[i];
		d_newShape.data[i] = newShape[i];
		d_starts.data[i] = starts[i];
		d_steps.data[i] = steps[i];
		totalElements *= newShape[i];
	}
	LAUNCH1(totalElements, slice_grad_kernel, gradOut, gradIn, d_origShape, d_newShape, d_starts, d_steps, rank, totalElements);
}

EXPORT void NativeConcat(const float** inputs, float* output, int numInputs,
	int outerSize, const int* concatSizes, int innerSize)
{
	if (inputs == nullptr || concatSizes == nullptr || numInputs <= 0) { g_lastError = cudaErrorInvalidValue; return; }
	int totalConcatSize = 0;
	for (int k = 0; k < numInputs; k++) totalConcatSize += concatSizes[k];

	int currentOffset = 0;
	g_lastError = cudaSuccess;
	for (int k = 0; k < numInputs; k++)
	{
		int inputConcatSize = concatSizes[k];
		long long totalElements = (long long)outerSize * inputConcatSize * innerSize;
		if (totalElements > 0)
		{
			int bs, gs;
			get_launch_config(totalElements, &bs, &gs);
			concat_copy_kernel << <gs, bs >> > (inputs[k], output, outerSize, inputConcatSize, totalConcatSize, innerSize, currentOffset);
			record_launch();
			if (g_lastError != cudaSuccess) return;
		}
		currentOffset += inputConcatSize;
	}
}

EXPORT void NativeConv2DForward(const float* input, const float* weight, float* output,
	int batch, int in_channels, int in_h, int in_w, int out_channels, int out_h, int out_w,
	int k_h, int k_w, int stride, int padding)
{
	LAUNCH1((long long)batch * out_channels * out_h * out_w, conv2d_forward_kernel,
		input, weight, output, batch, in_channels, in_h, in_w, out_channels, out_h, out_w, k_h, k_w, stride, padding);
}

EXPORT void NativeConv2DGradWeight(const float* input, const float* grad_out, float* grad_weight,
	int batch, int in_channels, int in_h, int in_w, int out_channels, int out_h, int out_w,
	int k_h, int k_w, int stride, int padding)
{
	LAUNCH1((long long)out_channels * in_channels * k_h * k_w, conv2d_grad_weight_kernel,
		input, grad_out, grad_weight, batch, in_channels, in_h, in_w, out_channels, out_h, out_w, k_h, k_w, stride, padding);
}

EXPORT void NativeConv2DGradInput(const float* grad_out, const float* weight, float* grad_input,
	int batch, int in_channels, int in_h, int in_w, int out_channels, int out_h, int out_w,
	int k_h, int k_w, int stride, int padding)
{
	LAUNCH1((long long)batch * in_channels * in_h * in_w, conv2d_grad_input_kernel,
		grad_out, weight, grad_input, batch, in_channels, in_h, in_w, out_channels, out_h, out_w, k_h, k_w, stride, padding);
}

EXPORT void NativeConv3DForward(const float* input, const float* weight, float* output,
	int batch, int in_channels, int in_d, int in_h, int in_w,
	int out_channels, int out_d, int out_h, int out_w,
	int k_d, int k_h, int k_w, int stride, int padding)
{
	LAUNCH1((long long)batch * out_channels * out_d * out_h * out_w, conv3d_forward_kernel,
		input, weight, output, batch, in_channels, in_d, in_h, in_w, out_channels, out_d, out_h, out_w, k_d, k_h, k_w, stride, padding);
}

EXPORT void NativeConv3DGradWeight(const float* input, const float* grad_out, float* grad_weight,
	int batch, int in_channels, int in_d, int in_h, int in_w,
	int out_channels, int out_d, int out_h, int out_w,
	int k_d, int k_h, int k_w, int stride, int padding)
{
	LAUNCH1((long long)out_channels * in_channels * k_d * k_h * k_w, conv3d_grad_weight_kernel,
		input, grad_out, grad_weight, batch, in_channels, in_d, in_h, in_w, out_channels, out_d, out_h, out_w, k_d, k_h, k_w, stride, padding);
}

EXPORT void NativeConv3DGradInput(const float* grad_out, const float* weight, float* grad_input,
	int batch, int in_channels, int in_d, int in_h, int in_w,
	int out_channels, int out_d, int out_h, int out_w,
	int k_d, int k_h, int k_w, int stride, int padding)
{
	LAUNCH1((long long)batch * in_channels * in_d * in_h * in_w, conv3d_grad_input_kernel,
		grad_out, weight, grad_input, batch, in_channels, in_d, in_h, in_w, out_channels, out_d, out_h, out_w, k_d, k_h, k_w, stride, padding);
}

EXPORT int InvokeHolonomicKernel(const ComplexDouble* inputs, const ComplexDouble* weights,
	const ComplexDouble* intWeights, ComplexDouble* outputs,
	int inputSize, int neuronCount, int fractalDepth)
{
	if (neuronCount <= 0) { g_lastError = cudaSuccess; return 0; }
	int bs = 256;
	int gs = (neuronCount + bs - 1) / bs;
	holonomic_kernel << <gs, bs >> > (inputs, weights, intWeights, outputs, inputSize, neuronCount, fractalDepth);
	record_launch();
	return (int)g_lastError;
}

EXPORT void NativeTopK(const float* input, float* outValues, float* outIndices, int outer, int dim, int inner, int k)
{
	if (outer <= 0 || dim <= 0 || inner <= 0 || k <= 0) { g_lastError = cudaSuccess; return; }
	long long idxCount = (long long)outer * k * inner;
	LAUNCH1(idxCount, set_scalar_kernel, outIndices, -1.0f, (int)idxCount);
	if (g_lastError != cudaSuccess) return;
	LAUNCH1((long long)outer * inner, topk_kernel, input, outValues, outIndices, outer, dim, inner, k);
}

EXPORT void NativeTopKScatterGrad(const float* gradOut, const float* indices, float* gradIn,
	int outer, int dim, int inner, int k)
{
	LAUNCH1((long long)outer * inner * k, topk_scatter_grad_kernel, gradOut, indices, gradIn, outer, dim, inner, k);
}

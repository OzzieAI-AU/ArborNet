#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

// Complex Tanh activation for Holonomic mapping
__device__ cuDoubleComplex complex_tanh(cuDoubleComplex z) {
    double x = cuCreal(z);
    double y = cuCimag(z);
    double denom = cosh(2.0 * x) + cos(2.0 * y);
    return make_cuDoubleComplex(sinh(2.0 * x) / denom, sin(2.0 * y) / denom);
}

extern "C" __global__ void HolonomicForwardKernel(
    const cuDoubleComplex* inputs,      // [inputSize]
    const cuDoubleComplex* weights,     // [neuronCount * inputSize]
    const cuDoubleComplex* intWeights,  // [neuronCount] (Internal Fractal Weights)
    cuDoubleComplex* outputs,           // [neuronCount]
    int inputSize,
    int neuronCount,
    int fractalDepth) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.y;
    if (idx >= neuronCount) return;

    // --- PHASE 1: HOLOGRAPHIC INTERFERENCE (Psi) ---
    // Every neuron thread aggregates the global wave state
    cuDoubleComplex psi = make_cuDoubleComplex(0.0, 0.0);
    
    for (int i = 0; i < inputSize; i++) {
        // Interference calculation: Z_j = sum(W_jk * X_k)
        // cuCmul handles the phase addition: (A1*A2) * e^(i(theta1+theta2))
        cuDoubleComplex interference = cuCmul(inputs[i], weights[idx * inputSize + i]);
        psi = cuCadd(psi, interference);
    }

    // --- PHASE 2: FRACTAL UNFOLDING (Z_t+1 = tanh(W_int * Z_t + Psi)) ---
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex w_int = intWeights[idx];

    for (int t = 0; t < fractalDepth; t++) {
        // Recursive loop: This is where the fractal geometry is "rendered" 
        // into the neuron's hidden state.
        cuDoubleComplex next_z = cuCadd(cuCmul(w_int, z), psi);
        z = complex_tanh(next_z);
    }

    // Map the final resonant state to output
    outputs[idx] = z;
}
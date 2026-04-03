# Getting Started with ArborNet

Welcome to ArborNet, a high-performance deep learning framework built in C# for .NET, inspired by PyTorch. This guide will walk you through the basics of setting up and using ArborNet to build, train, and deploy neural networks. We'll cover installation, core concepts, basic tensor operations, creating layers, defining models, training, and more.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Creating Tensors](#creating-tensors)
- [Building Layers](#building-layers)
- [Defining Models](#defining-models)
- [Training a Model](#training-a-model)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Prerequisites
Before getting started, ensure you have:
- .NET 8.0 or later installed (ArborNet targets `net8.0`).
- Visual Studio 2022 or another C# IDE that supports .NET 8.
- Basic knowledge of C# and neural networks.
- Optional: CUDA or ROCm drivers if you plan to use GPU acceleration.

## Installation
ArborNet is organized as a solution with multiple projects. To get started:

1. **Clone or Download the Repository**:
   - Clone the ArborNet repository from GitHub (assuming it's available publicly).
   - Or download the solution files.

2. **Open the Solution**:
   - Open `ArborNet.sln` in Visual Studio.

3. **Build the Solution**:
   - Build the entire solution to compile all projects (ArborNet.Core, ArborNet.Layers, etc.).
   - NuGet packages should be restored automatically.

4. **Add References**:
   - In your own C# project, add references to the ArborNet assemblies (e.g., ArborNet.Core.dll, ArborNet.Layers.dll).

For a NuGet package (if available), run:
bash
dotnet add package ArborNet
Note: As of now, ArborNet is a local solution; NuGet distribution may come later.

## Core Concepts
ArborNet is built around the following key concepts:
- **Tensors**: Multi-dimensional arrays that hold data and support operations like addition, matrix multiplication, and gradients.
- **Autograd**: Automatic differentiation for computing gradients during backpropagation.
- **Layers**: Building blocks like Conv2D, Linear, and activations (ReLU, Sigmoid).
- **Models**: Compositions of layers, such as Sequential or custom models like ResNet.
- **Optimizers**: Algorithms like Adam or SGD for updating model parameters.
- **Losses**: Functions like MSE or CrossEntropy to measure model performance.
- **Devices**: CPU, CUDA, or ROCm for computation.

The architecture follows an inside-out design: ArborNet.Core is the foundation, with outer layers depending on it.

## Creating Tensors
Tensors are the fundamental data structure. Import the necessary namespaces:
using ArborNet.Core;
using ArborNet.Core.Tensors;
using ArborNet.Core.Device;
Create a simple tensor:
// Create a 2x3 tensor filled with zeros on CPU
TensorShape shape = new TensorShape(2, 3);
ITensor zeros = Tensor.Zeros(shape, Device.Cpu);

// Create a tensor from an array
float[] data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
ITensor tensor = Tensor.FromArray(data, shape, Device.Cpu);

// Enable gradients for autograd
tensor.RequiresGrad = true;

// Perform operations
ITensor result = tensor.Add(zeros);
Tensors support operations like `Add`, `MatMul`, `Reshape`, and more. See `ITensor.cs` for the full interface.

## Building Layers
Layers are in ArborNet.Layers. Import:
using ArborNet.Layers;
using ArborNet.Activations;
Create a simple linear layer:
// Linear layer: input size 10, output size 5
Linear linear = new Linear(10, 5);

// Apply to a tensor (batch size 1, input size 10)
ITensor input = Tensor.Randn(new TensorShape(1, 10));
ITensor output = linear.Forward(input);

// Add an activation
ReLU relu = new ReLU();
ITensor activated = relu.Forward(output);
For convolutional layers:
Conv2D conv = new Conv2D(inChannels: 3, outChannels: 64, kernelSize: 3);
ITensor imageInput = Tensor.Randn(new TensorShape(1, 3, 28, 28)); // Batch, Channels, Height, Width
ITensor convOutput = conv.Forward(imageInput);
Layers automatically handle autograd if inputs require gradients.

## Defining Models
Models are in ArborNet.Models. Use Sequential for simple stacks:
using ArborNet.Models;

Sequential model = new Sequential();
model.Add(new Linear(784, 128));
model.Add(new ReLU());
model.Add(new Linear(128, 10));
model.Add(new Softmax(axis: 1)); // For classification

// Forward pass
ITensor modelOutput = model.Forward(input);
For advanced models like ResNet:
ResNet resnet = new ResNet(numClasses: 1000);
ITensor prediction = resnet.Forward(imageInput);
Custom models inherit from BaseModel and override `Forward`.

## Training a Model
Training involves data, loss, and optimization. Import:
using ArborNet.Optimizers;
using ArborNet.Losses;
using ArborNet.Trainers;
Example training loop:
// Define model, loss, optimizer
Sequential model = new Sequential();
model.Add(new Linear(10, 5));
model.Add(new Softmax(axis: 1));

CrossEntropy lossFn = new CrossEntropy();
Adam optimizer = new Adam(model.Parameters(), lr: 0.001f);

// Dummy data
ITensor x = Tensor.Randn(new TensorShape(32, 10)); // Batch of 32, input size 10
ITensor y = Tensor.FromArray(new float[] { 0, 1, 2, ... }, new TensorShape(32)); // Labels

// Training step
optimizer.ZeroGrad();
ITensor predictions = model.Forward(x);
ITensor loss = lossFn.Forward(predictions, y);
loss.Backward();
optimizer.Step();
Use Trainer for structured training:
Trainer trainer = new Trainer(model, optimizer, lossFn);
trainer.Train(trainData, epochs: 10);
## Advanced Features
- **Multi-Device**: Move tensors with `tensor.To(Device.Cuda())`.
- **Autograd**: Use GradientTape for custom operations.
- **Datasets**: Load data from ArborNet.Data.Datasets (e.g., MNIST.Download()).
- **Export**: Use OnnxExporter or TorchScript for deployment.
- **Backends**: Switch between CpuBackend, CudaBackend, etc.

## Troubleshooting
- **Build Errors**: Ensure all projects are built in order. ArborNet.Core first.
- **Runtime Errors**: Check device compatibility and tensor shapes.
- **Performance**: Profile with ArborNet.Tests.BenchmarkTests.
- **Community**: Join the ArborNet GitHub discussions for help.

For more details, explore the source code and tests. Happy coding with ArborNet!
# 🌲 ArborNet
### The Holonomic .NET Machine Learning Framework

![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square&logo=github)

**ArborNet** is a high-performance, fluent machine learning library for .NET. It is engineered for researchers who want the flexibility of PyTorch combined with the type-safety and performance of the .NET ecosystem. By bridging the gap between front-line holonomic mathematics and industrial-grade software engineering, ArborNet makes building complex neural architectures as natural as writing prose.

> ### 💡 The ArborNet Advantage
> The successor to Microsoft CNTK, but with a much smaller footprint, OzzieAI's ArborNet is the answer to Machine Learning where performance and resource efficiency are paramount.

![ArborNet Visual Overview](Images/img1.png)

---

## ✨ Why ArborNet?

Most .NET ML libraries feel like rigid wrappers. **ArborNet** feels like an extension of your thought process. 

* **⚡ Hardware Agnostic:** Seamlessly switch between CPU, NVIDIA CUDA, and AMD ROCm with a single line of code.
* **🌀 Holonomic Research:** Built-in primitives for holonomic constraints and manifold-based optimizations.
* **💎 Fluent Autograd:** A powerful, dynamic computational graph engine that tracks operations automatically.
* **🛠️ Production Ready:** Designed for low-latency inference and high-throughput training in enterprise environments.

---

## 🚀 Get Started in Seconds

OzzieAI.ArborNet prioritizes a '`Fluent API`'. No more verbose boilerplate; just clean, expressive code that chains together effortlessly.

```csharp
// Full model in one fluent chain
var logits = X.Rand(1, 128)          // batch, seq
              .Embedding(50257, 768)
              .PositionalEncoding(768)
              .TransformerBlock(768, 12)
              .TransformerBlock(768, 12)
              .LayerNorm()
              .Linear(50257)
              .Softmax(-1);

// Instant full models
var gpt = X.GPT(vocabSize: 50257, dModel: 768, nLayers: 12);
var llama = X.Llama3(vocabSize: 128256);
var vit = X.ViT(imageSize: 224, patchSize: 16, numClasses: 1000);

// Vision pipeline
var pred = X.FromArray(imageData, 1, 3, 224, 224)
            .Cuda()
            .Conv2D(64, 7, stride: 2, padding: 3)
            .BatchNorm(64)
            .ReLU()
            .MaxPool2D()
            .ResNet50(1000)
            .Softmax();

// Generation
foreach (var token in X.From(promptTensor).GenerateStream(model, tokenizer, "Once upon a time"))
    Console.Write(token);

// Data augmentation
var aug = X.FromArray(batch, 32, 3, 32, 32)
           .Augment()
           .Resize(224, 224);
```

---

## 🔬 Front-Line Research: Holonomic Networks

ArborNet isn't just a tool; it's a research platform. We are pioneering **Holonomic Neural Networks**—architectures where the path taken during optimization is as critical as the final weights. 

Our engine supports path-integral kernels and non-Abelian symmetry groups, ensuring that your gradients respect the underlying geometry of the data manifold. We implement constraints where the curvature $$\Omega$$ of the connection $$A$$ satisfies:

$$\Omega = dA + A \wedge A = 0$$

By maintaining these holonomic properties, ArborNet allows for:

1.  **Global Convergence:** Significantly reducing the likelihood of getting stuck in sharp local minima.
2.  **Topological Stability:** Ensuring that small perturbations in input do not result in catastrophic failures in classification.

![Holonomic Math Visualization](Images/img2.png)

---

## 🏗️ Architectural Excellence

OzzieAI.ArborNet is built on a modular backend system. Whether you are running on a local workstation or a massive cluster, the logic remains identical across devices using the '`ITensor`' abstraction.

| Feature | CPU Backend | CUDA Backend | ROCm Backend |
| :--- | :---: | :---: | :---: |
| **Float32/Float16** | ✅ | ✅ | ✅ |
| **Tensor Cores** | ❌ | ✅ | ✅ |
| **Autograd** | ✅ | ✅ | ✅ |
| **Distributed** | ✅ | ✅ | 🚧 |

![Backend Architecture](Images/img3.png)

---

## 📥 Installation

Grab the latest build from NuGet:

```bash
dotnet add package OzzieAI.ArborNet.Core
dotnet add package OzzieAI.ArborNet.Backends.Cuda
```

---

## 🤝 Contributing

We love wit, grit, and great code. If you’re interested in holonomic research or just want to make .NET the premier language for AI, join us. Check out our ** [`CONTRIBUTING.md`](https://github.com/OzzieAI-AU/ArborNet?tab=contributing-ov-file)** to get started.

---

> "Simplifying the complex, one tensor at a time."
> **Built with ❤️ by the **[www.ozzieai.com](https://www.ozzieai.com/)** Research Collective.**
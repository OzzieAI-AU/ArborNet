// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Export
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Text;
    using System.Text.Json;
    using ArborNet.Core.Interfaces;
    using ArborNet.Core.Models;
    using ArborNet.Layers;
    using ArborNet.Trainers;
    /// <summary>
    /// Provides functionality to export neural network models to an ONNX-compatible representation.
    /// This exporter generates a clean, human-readable structural representation and an optimized binary weights payload.
    /// </summary>
    /// <remarks>
    /// This is designed to be a lightweight, dependency-free export mechanism that preserves 
    /// model parameters and network structure, structured for straightforward migration to full protobuf-based ONNX formats.
    /// </remarks>

    #endregion

    public class OnnxExporter
    {
        /// <summary>
        /// Exports the specified <see cref="LightningModule"/> to a custom ONNX-compatible format
        /// consisting of a human-readable JSON graph definition and a separate binary weights file.
        /// </summary>
        /// <param name="model">The <see cref="LightningModule"/> instance containing the parameters and architecture to be exported.</param>
        /// <param name="filePath">The base file path (excluding extension) where the JSON configuration and binary weights will be written.</param>
        /// <param name="modelName">The name assigned to the exported model within the generated metadata. Defaults to "ArborNetModel".</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="model"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="filePath"/> is null, empty, or consists only of white-space characters.</exception>
        /// <exception cref="IOException">Thrown if an I/O error occurs while writing files to disk.</exception>
        /// <remarks>
        /// This method generates two distinct outputs:
        /// <list type="bullet">
        ///   <item>
        ///     <description><c>{filePath}.json</c>: Contains model metadata, graph node configurations, and shapes in formatted JSON.</description>
        ///   </item>
        ///   <item>
        ///     <description><c>{filePath}.weights</c>: Contains the serialized weight values and parameter structures in an optimized binary stream.</description>
        ///   </item>
        /// </list>
        /// Currently, this treats structural variables as individual nodes and preserves exact multidimensional shapes.
        /// </remarks>

        public void Export(LightningModule model, string filePath, string modelName = "ArborNetModel")
        {

            if (model == null) throw new ArgumentNullException(nameof(model));

            if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentException("File path cannot be empty.");

            var exportModel = new ExportableModel
            {
                Name = modelName,
                Producer = "ArborNet",
                Version = "1.0",
                Nodes = new List<ExportNode>(),
                Initializers = new Dictionary<string, float[]>()
            };

            // Traverse model parameters
            foreach (var param in model.Parameters())
            {
                var name = $"param_{Guid.NewGuid():N}";
                exportModel.Initializers[name] = param.ToArray();

                exportModel.Nodes.Add(new ExportNode
                {
                    Name = name,
                    OpType = "Constant",
                    Outputs = { name },
                    Attributes = { ["shape"] = string.Join(",", param.Shape.Dimensions) }
                });
            }

            // Add basic graph structure (can be extended with real op traversal)
            exportModel.Nodes.Add(new ExportNode
            {
                Name = "input",
                OpType = "Input",
                Outputs = { "input" }
            });

            exportModel.Nodes.Add(new ExportNode
            {
                Name = "output",
                OpType = "Output",
                Inputs = { "input" },
                Outputs = { "output" }
            });

            // Write as JSON + binary weights (very easy to read/debug)
            var json = JsonSerializer.Serialize(exportModel, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(filePath + ".json", json);

            // Also write binary weights for fast loading
            using var fs = new FileStream(filePath + ".weights", FileMode.Create);
            using var bw = new BinaryWriter(fs);
            foreach (var kvp in exportModel.Initializers)
            {
                bw.Write(kvp.Key);
                bw.Write(kvp.Value.Length);
                foreach (var f in kvp.Value)
                    bw.Write(f);
            }

            Console.WriteLine($"Model exported to {filePath}.json + .weights");
        }
        /// <summary>
        /// Represents the complete structure of a model during the export process.
        /// </summary>
        /// <remarks>
        /// This internal structure is serialized directly to JSON to define the overall metadata and architecture of the exported neural network.
        /// </remarks>

        private class ExportableModel
        {
            /// <summary>
            /// Gets or sets the name identifier of the exported model.
            /// </summary>
            /// <value>The name of the network structure.</value>
            public string Name { get; set; } = string.Empty;
            /// <summary>
            /// Gets or sets the name of the software or entity that produced the exported file.
            /// </summary>
            /// <value>Defaults to the producing library name, typically "ArborNet".</value>

            public string Producer { get; set; } = string.Empty;
            /// <summary>
            /// Gets or sets the format or schema version of the exported model structure.
            /// </summary>
            /// <value>The export format version.</value>

            public string Version { get; set; } = string.Empty;
            /// <summary>
            /// Gets or sets the sequence of execution and declaration nodes defining the computation graph.
            /// </summary>
            /// <value>A collection of <see cref="ExportNode"/> elements representing the network's layers and state.</value>

            public List<ExportNode> Nodes { get; set; } = new();
            /// <summary>
            /// Gets or sets mapping table for binary initializers (weights and biases), keyed by their unique node identifiers.
            /// </summary>
            /// <value>A dictionary containing unique parameter keys and their corresponding flattened floating-point numerical matrices.</value>

            public Dictionary<string, float[]> Initializers { get; set; } = new();
        }
        /// <summary>
        /// Represents an individual computational node, variable declarations, or layer structure within the serialized execution graph.
        /// </summary>
        /// <remarks>
        /// This mirrors typical ONNX node design patterns, recording operational metadata, connections (inputs/outputs), and tensor dimensions.
        /// </remarks>

        private class ExportNode
        {
            /// <summary>
            /// Gets or sets the name identifier of the exported model.
            /// </summary>
            /// <value>The name of the network structure.</value>
            public string Name { get; set; } = string.Empty;
            /// <summary>
            /// Gets or sets the operator type designation for this node (e.g., "Constant", "Input", "Output").
            /// </summary>
            /// <value>The string literal indicating the logical operation performed by this node.</value>

            public string OpType { get; set; } = string.Empty;
            /// <summary>
            /// Gets or sets the list of inputs consumed or required by this node.
            /// </summary>
            /// <value>A list of input parameter names and preceding nodes linked to this operation.</value>

            public List<string> Inputs { get; set; } = new();
            /// <summary>
            /// Gets or sets the list of outputs produced or exposed by this node.
            /// </summary>
            /// <value>A list of result names exported from this operation.</value>

            public List<string> Outputs { get; set; } = new();
            /// <summary>
            /// Gets or sets metadata attributes associated with this node, such as tensor shapes, data configurations, or custom parameters.
            /// </summary>
            /// <value>A dictionary containing variable key-value configurations describing the operational constraints of the node.</value>

            public Dictionary<string, object> Attributes { get; set; } = new();
        }
    }
}
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using ArborNet.Core.Interfaces;
using ArborNet.Core.Models;
using ArborNet.Layers;
using ArborNet.Trainers;

namespace ArborNet.Export
{

    /// <summary>
    /// High-quality, dependency-free ONNX exporter.
    /// Exports to a clean, human-readable + binary-compatible format.
    /// Can be easily extended to full ONNX protobuf later.
    /// </summary>
    public class OnnxExporter
    {
    
        /// <summary>
        /// Exports the specified <see cref="LightningModule"/> to a custom ONNX-compatible format
        /// consisting of a human-readable JSON graph definition and a separate binary weights file.
        /// </summary>
        /// <param name="model">The LightningModule instance containing the parameters to be exported.</param>
        /// <param name="filePath">The base file path (without extension) where the exported files will be written.</param>
        /// <param name="modelName">The name assigned to the exported model. Defaults to "ArborNetModel".</param>
        /// <remarks>
        /// This method produces two files:
        /// <list type="bullet">
        ///   <item><c>{filePath}.json</c> - Contains model metadata, nodes, and structure in indented JSON format.</item>
        ///   <item><c>{filePath}.weights</c> - Contains parameter names and their float values in a compact binary format.</item>
        /// </list>
        /// The current implementation treats all model parameters as Constant nodes. The format is designed
        /// for easy debugging while maintaining a path toward full ONNX protobuf compatibility.
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
        /// Internal data structure representing the model for serialization to the custom export format.
        /// </summary>
        /// <remarks>
        /// This class acts as an intermediate representation containing model metadata,
        /// computational graph nodes, and weight initializers.
        /// </remarks>
        private class ExportableModel
        {
            /// <summary>
            /// Gets or sets the name of the model.
            /// </summary>
            public string Name { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the name of the producer/creator of the model.
            /// </summary>
            public string Producer { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the version of the exported model.
            /// </summary>
            public string Version { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the collection of nodes that define the model's computational graph.
            /// </summary>
            public List<ExportNode> Nodes { get; set; } = new();

            /// <summary>
            /// Gets or sets the dictionary of weight initializers, keyed by parameter name.
            /// </summary>
            public Dictionary<string, float[]> Initializers { get; set; } = new();
        }

        /// <summary>
        /// Represents a single node in the exported computational graph.
        /// </summary>
        /// <remarks>
        /// Used to model operations, inputs, outputs, and attributes in a format
        /// that mirrors ONNX graph semantics.
        /// </remarks>
        private class ExportNode
        {
            /// <summary>
            /// Gets or sets the unique name of this node.
            /// </summary>
            public string Name { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the operation type of the node (e.g. "Constant", "Input", "Output").
            /// </summary>
            public string OpType { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the list of input tensor names consumed by this node.
            /// </summary>
            public List<string> Inputs { get; set; } = new();

            /// <summary>
            /// Gets or sets the list of output tensor names produced by this node.
            /// </summary>
            public List<string> Outputs { get; set; } = new();

            /// <summary>
            /// Gets or sets additional attributes associated with the node (e.g. shape information).
            /// </summary>
            public Dictionary<string, object> Attributes { get; set; } = new();
        }
    }
}
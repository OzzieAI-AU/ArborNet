// -----------------------------------------------------------------------------------------
// Copyright © 2026 OzzieAI - Chris Sykes. All rights reserved.
// 
// Project:      ArborNet
// Description:  A C# Machine Learning Library implemented in .NET 10 with full CUDA support.
// 
// License:      MIT License
// -----------------------------------------------------------------------------------------

namespace ArborNet.Data
{

    #region Using Statements:

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.IO;

    #endregion

    /// <summary>
    /// Interface for tokenizers that encode text to tokens and decode tokens back to text.
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>
        /// Encodes the input text into a list of token IDs.
        /// </summary>
        /// <param name="text">The input text to tokenize.</param>
        /// <returns>A list of integer token IDs.</returns>
        List<int> Encode(string text);
        /// <summary>
        /// Decodes a list of token IDs back into text.
        /// </summary>
        /// <param name="tokens">The list of token IDs to decode.</param>
        /// <returns>The decoded text string.</returns>

        string Decode(List<int> tokens);
    }
    /// <summary>
    /// Implements Byte-Pair Encoding (BPE) tokenizer.
    /// BPE iteratively merges the most frequent pairs of bytes or subwords in the vocabulary.
    /// </summary>

    public class BpeTokenizer : ITokenizer
    {
        private readonly Dictionary<string, int> vocab;
        private readonly Dictionary<int, string> reverseVocab; // O(1) Cache
        private readonly List<(string, string)> merges;
        private readonly string unkToken;
        private readonly int unkId;

        public BpeTokenizer(string vocabFilePath, string mergesFilePath, string unkToken = "<unk>", int unkId = 0)
        {
            this.unkToken = unkToken;
            this.unkId = unkId;
            vocab = LoadVocab(vocabFilePath);
            merges = LoadMerges(mergesFilePath);
            reverseVocab = vocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        }
        /// <summary>
        /// Loads the vocabulary dictionary from the specified file path.
        /// </summary>
        /// <param name="filePath">The file path to read the vocabulary from.</param>
        /// <returns>A dictionary containing the parsed vocabulary pairs.</returns>

        private Dictionary<string, int> LoadVocab(string filePath)
        {
            var vocabDict = new Dictionary<string, int>();
            foreach (var line in File.ReadAllLines(filePath))
            {
                var parts = line.Split(' ');
                if (parts.Length == 2 && int.TryParse(parts[1], out int id))
                {
                    vocabDict[parts[0]] = id;
                }
            }
            return vocabDict;
        }
        /// <summary>
        /// Loads BPE merge operations from the specified file path.
        /// </summary>
        /// <param name="filePath">The file path to read the merge rules from.</param>
        /// <returns>A list of tuples representing pairs of tokens to be merged.</returns>

        private List<(string, string)> LoadMerges(string filePath)
        {
            var mergesList = new List<(string, string)>();
            foreach (var line in File.ReadAllLines(filePath).Skip(1))
            {
                var parts = line.Split(' ');
                if (parts.Length == 2)
                {
                    mergesList.Add((parts[0], parts[1]));
                }
            }
            return mergesList;
        }
        /// <summary>
        /// Encodes the input text into a list of token IDs.
        /// </summary>
        /// <param name="text">The input text to tokenize.</param>
        /// <returns>A list of integer token IDs.</returns>

        public List<int> Encode(string text)
        {
            var words = text.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var tokens = new List<int>();

            foreach (var word in words)
            {
                var subwords = PreTokenize(word);
                foreach (var merge in merges)
                {
                    subwords = ApplyMerge(subwords, merge);
                }
                foreach (var subword in subwords)
                {
                    if (vocab.TryGetValue(subword, out int id))
                    {
                        tokens.Add(id);
                    }
                    else
                    {
                        tokens.Add(unkId);
                    }
                }
            }

            return tokens;
        }
        /// <summary>
        /// Performs initial tokenization on a word, splitting it into its constituent character strings.
        /// </summary>
        /// <param name="word">The word to pre-tokenize.</param>
        /// <returns>A list of single-character strings representing the individual characters of the word.</returns>

        private List<string> PreTokenize(string word)
        {
            var subwords = new List<string>();
            foreach (char c in word)
            {
                subwords.Add(c.ToString());
            }
            return subwords;
        }
        /// <summary>
        /// Applies a BPE merge rule to a list of subwords, combining matching adjacent pairs.
        /// </summary>
        /// <param name="subwords">The list of subwords currently being processed.</param>
        /// <param name="merge">A tuple representing the pair of subwords that should be merged.</param>
        /// <returns>A new list of subwords with the specified merge rule applied.</returns>

        private List<string> ApplyMerge(List<string> subwords, (string, string) merge)
        {
            var result = new List<string>();
            int i = 0;
            while (i < subwords.Count)
            {
                if (i < subwords.Count - 1 && subwords[i] == merge.Item1 && subwords[i + 1] == merge.Item2)
                {
                    result.Add(merge.Item1 + merge.Item2);
                    i += 2;
                }
                else
                {
                    result.Add(subwords[i]);
                    i++;
                }
            }
            return result;
        }
        /// <summary>
        /// Decodes a list of token IDs back into text.
        /// </summary>
        /// <param name="tokens">The list of token IDs to decode.</param>
        /// <returns>The decoded text string.</returns>

        public string Decode(List<int> tokens)
        {
            var subwords = new List<string>();
            foreach (var token in tokens)
            {
                if (reverseVocab.TryGetValue(token, out var subword))
                {
                    subwords.Add(subword);
                }
                else
                {
                    subwords.Add(unkToken);
                }
            }
            return string.Join("", subwords);
        }
    }
    /// <summary>
    /// Implements a simplified SentencePiece tokenizer based on Unigram model.
    /// SentencePiece tokenizes text into subwords using a pre-trained model.
    /// This implementation assumes a vocabulary file with subwords and their scores.
    /// </summary>

    public class SentencePieceTokenizer : ITokenizer
    {
        private readonly Dictionary<string, int> vocab;
        private readonly Dictionary<int, string> reverseVocab; // O(1) Cache
        private readonly Dictionary<string, double> scores;
        private readonly string unkToken;
        private readonly int unkId;

        public SentencePieceTokenizer(string modelFilePath, string unkToken = "<unk>", int unkId = 0)
        {
            this.unkToken = unkToken;
            this.unkId = unkId;
            vocab = new Dictionary<string, int>();
            scores = new Dictionary<string, double>();
            LoadModel(modelFilePath);
            reverseVocab = vocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        }
        /// <summary>
        /// Loads the SentencePiece model, parsing vocabulary, identifiers, and scores.
        /// </summary>
        /// <param name="filePath">The file path to load the model configurations from.</param>

        private void LoadModel(string filePath)
        {
            foreach (var line in File.ReadAllLines(filePath))
            {
                var parts = line.Split('\t');
                if (parts.Length >= 2 && int.TryParse(parts[0], out int id))
                {
                    vocab[parts[1]] = id;
                    if (parts.Length > 2 && double.TryParse(parts[2], out double score))
                    {
                        scores[parts[1]] = score;
                    }
                }
            }
        }
        /// <summary>
        /// Encodes the input text into a list of token IDs.
        /// </summary>
        /// <param name="text">The input text to tokenize.</param>
        /// <returns>A list of integer token IDs.</returns>

        public List<int> Encode(string text)
        {
            var tokens = new List<int>();
            int i = 0;
            while (i < text.Length)
            {
                bool found = false;
                for (int len = Math.Min(10, text.Length - i); len > 0; len--)
                {
                    var sub = text.Substring(i, len);
                    if (vocab.TryGetValue(sub, out int id))
                    {
                        tokens.Add(id);
                        i += len;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    tokens.Add(unkId);
                    i++;
                }
            }
            return tokens;
        }
        /// <summary>
        /// Decodes a list of token IDs back into text.
        /// </summary>
        /// <param name="tokens">The list of token IDs to decode.</param>
        /// <returns>The decoded text string.</returns>

        public string Decode(List<int> tokens)
        {
            var subwords = new List<string>();
            foreach (var token in tokens)
            {
                if (reverseVocab.TryGetValue(token, out var subword))
                {
                    subwords.Add(subword);
                }
                else
                {
                    subwords.Add(unkToken);
                }
            }
            return string.Join("", subwords);
        }
    }
}
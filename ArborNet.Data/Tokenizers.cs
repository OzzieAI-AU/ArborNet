using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace ArborNet.Data
{
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
        /// <summary>
        /// Vocabulary dictionary mapping subword tokens to their corresponding integer IDs.
        /// </summary>
        private readonly Dictionary<string, int> vocab;

        /// <summary>
        /// Ordered list of merge operations. Each tuple represents a pair of tokens 
        /// to be merged, in the order of priority.
        /// </summary>
        private readonly List<(string, string)> merges;

        /// <summary>
        /// The string representation of the unknown token.
        /// </summary>
        private readonly string unkToken;

        /// <summary>
        /// The integer ID assigned to the unknown token.
        /// </summary>
        private readonly int unkId;

        /// <summary>
        /// Initializes a new instance of the BpeTokenizer.
        /// </summary>
        /// <param name="vocabFilePath">Path to the vocabulary file (key-value pairs of subword to ID).</param>
        /// <param name="mergesFilePath">Path to the merges file (list of pairs to merge).</param>
        /// <param name="unkToken">Unknown token string.</param>
        /// <param name="unkId">Unknown token ID.</param>
        public BpeTokenizer(string vocabFilePath, string mergesFilePath, string unkToken = "<unk>", int unkId = 0)
        {
            this.unkToken = unkToken;
            this.unkId = unkId;
            vocab = LoadVocab(vocabFilePath);
            merges = LoadMerges(mergesFilePath);
        }

        /// <summary>
        /// Loads the vocabulary from a file where each line contains a token 
        /// followed by its ID, separated by a space.
        /// </summary>
        /// <param name="filePath">The path to the vocabulary file.</param>
        /// <returns>A dictionary mapping tokens to their IDs.</returns>
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
        /// Loads the merge rules from a file. The first line is skipped as it is assumed to be a header.
        /// </summary>
        /// <param name="filePath">The path to the merges file.</param>
        /// <returns>A list of merge pairs in the order they should be applied.</returns>
        private List<(string, string)> LoadMerges(string filePath)
        {
            var mergesList = new List<(string, string)>();
            foreach (var line in File.ReadAllLines(filePath).Skip(1)) // Skip header
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
        /// Encodes the input text using BPE.
        /// </summary>
        /// <param name="text">The input text.</param>
        /// <returns>List of token IDs.</returns>
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
        /// Splits a word into a list of individual characters to serve as the initial subwords 
        /// for the BPE merging process.
        /// </summary>
        /// <param name="word">The word to pre-tokenize.</param>
        /// <returns>A list of single-character strings.</returns>
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
        /// Applies a single merge operation to the list of subwords by combining 
        /// adjacent matching pairs.
        /// </summary>
        /// <param name="subwords">The current list of subwords.</param>
        /// <param name="merge">The pair of subwords that should be merged.</param>
        /// <returns>A new list of subwords with the merge applied where possible.</returns>
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
        /// Decodes the list of token IDs back to text.
        /// </summary>
        /// <param name="tokens">List of token IDs.</param>
        /// <returns>The decoded text.</returns>
        public string Decode(List<int> tokens)
        {
            var subwords = new List<string>();
            foreach (var token in tokens)
            {
                var subword = vocab.FirstOrDefault(kvp => kvp.Value == token).Key ?? unkToken;
                subwords.Add(subword);
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
        /// <summary>
        /// Maps subword pieces to their corresponding integer token IDs.
        /// </summary>
        private readonly Dictionary<string, int> vocab;

        /// <summary>
        /// Maps subword pieces to their associated scores (loaded for completeness 
        /// but not used in the current simplified greedy implementation).
        /// </summary>
        private readonly Dictionary<string, double> scores;

        /// <summary>
        /// The string representation of the unknown token.
        /// </summary>
        private readonly string unkToken;

        /// <summary>
        /// The integer ID for the unknown token.
        /// </summary>
        private readonly int unkId;

        /// <summary>
        /// Initializes a new instance of the SentencePieceTokenizer.
        /// </summary>
        /// <param name="modelFilePath">Path to the SentencePiece model file (subword to ID and score).</param>
        /// <param name="unkToken">Unknown token string.</param>
        /// <param name="unkId">Unknown token ID.</param>
        public SentencePieceTokenizer(string modelFilePath, string unkToken = "<unk>", int unkId = 0)
        {
            this.unkToken = unkToken;
            this.unkId = unkId;
            vocab = new Dictionary<string, int>();
            scores = new Dictionary<string, double>();
            LoadModel(modelFilePath);
        }

        /// <summary>
        /// Loads the SentencePiece model from a file. Each line is expected to be 
        /// tab-separated containing ID, subword, and optionally a score.
        /// </summary>
        /// <param name="filePath">Path to the model file.</param>
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
        /// Encodes the input text using SentencePiece (simplified greedy decoding).
        /// </summary>
        /// <param name="text">The input text.</param>
        /// <returns>List of token IDs.</returns>
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
        /// Decodes the list of token IDs back to text.
        /// </summary>
        /// <param name="tokens">List of token IDs.</param>
        /// <returns>The decoded text.</returns>
        public string Decode(List<int> tokens)
        {
            var subwords = new List<string>();
            foreach (var token in tokens)
            {
                var subword = vocab.FirstOrDefault(kvp => kvp.Value == token).Key ?? unkToken;
                subwords.Add(subword);
            }
            return string.Join("", subwords);
        }
    }
}
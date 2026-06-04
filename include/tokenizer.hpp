#pragma once
// =============================================================================
// tokenizer.hpp — Character-level tokenizer
// =============================================================================
// A minimal tokenizer that operates at the character level.
// Each unique character in the training corpus becomes one token.
//
// Design choices for this educational project:
//   • Character-level: zero external dependencies, simple encode/decode.
//   • Fixed special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3.
//   • Vocabulary is built by calling build_vocab(text) or add_char(c).
//
// Real tokenizers (BPE, WordPiece, SentencePiece) operate at sub-word level
// for better compression; the interface is identical.
// =============================================================================

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

class Tokenizer {
public:
    // Special token IDs — always present
    static constexpr int PAD_ID = 0;   // padding (for batching)
    static constexpr int UNK_ID = 1;   // unknown character
    static constexpr int BOS_ID = 2;   // beginning-of-sequence
    static constexpr int EOS_ID = 3;   // end-of-sequence

    // ── Constructor ───────────────────────────────────────────────────────────
    // Initialises the vocabulary with the four special tokens.
    Tokenizer();

    // ── Vocabulary construction ───────────────────────────────────────────────
    // Scan text and register every unique character that is not already known.
    void build_vocab(const std::string& text);

    // Add a single character to the vocabulary (no-op if already present).
    void add_char(char c);

    // ── Encoding ──────────────────────────────────────────────────────────────
    // Convert a string to a sequence of token IDs.
    // Unknown characters map to UNK_ID.
    // If add_bos is true, prepend BOS_ID.
    // If add_eos is true, append  EOS_ID.
    std::vector<int> encode(const std::string& text,
                            bool add_bos = true,
                            bool add_eos = false) const;

    // ── Decoding ──────────────────────────────────────────────────────────────
    // Convert a sequence of token IDs back to a string.
    // Special tokens (PAD, BOS, EOS) are omitted from the output.
    // UNK_ID renders as '?'.
    std::string decode(const std::vector<int>& ids) const;

    // ── Lookup helpers ────────────────────────────────────────────────────────
    int         char_to_id(char c)   const;
    std::string id_to_str(int id)    const;
    bool        has_char(char c)     const;

    // ── Info ──────────────────────────────────────────────────────────────────
    int  vocab_size()  const { return static_cast<int>(id_to_char_.size()); }
    void print_vocab() const;

private:
    std::unordered_map<char, int> char_to_id_;   // char → token ID
    std::vector<std::string>      id_to_char_;   // token ID → display string
};

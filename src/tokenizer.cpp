// =============================================================================
// tokenizer.cpp — Character-level tokenizer implementation
// =============================================================================
#include "tokenizer.hpp"
#include <iostream>
#include <stdexcept>

// =============================================================================
// Constructor — register the four special tokens
// =============================================================================
Tokenizer::Tokenizer() {
    // ID 0: <PAD>
    id_to_char_.push_back("<PAD>");
    // ID 1: <UNK>
    id_to_char_.push_back("<UNK>");
    // ID 2: <BOS>
    id_to_char_.push_back("<BOS>");
    // ID 3: <EOS>
    id_to_char_.push_back("<EOS>");
    // char_to_id_ only maps real chars; special tokens are handled by their IDs
}

// =============================================================================
// build_vocab — scan text and register unique characters
// =============================================================================
void Tokenizer::build_vocab(const std::string& text) {
    for (char c : text) add_char(c);
}

// =============================================================================
// add_char — add a single character if not already present
// =============================================================================
void Tokenizer::add_char(char c) {
    if (char_to_id_.count(c) == 0) {
        int new_id = static_cast<int>(id_to_char_.size());
        char_to_id_[c] = new_id;
        id_to_char_.push_back(std::string(1, c));
    }
}

// =============================================================================
// encode — text → token IDs
// =============================================================================
std::vector<int> Tokenizer::encode(const std::string& text,
                                   bool add_bos,
                                   bool add_eos) const
{
    std::vector<int> ids;
    ids.reserve(text.size() + 2);

    if (add_bos) ids.push_back(BOS_ID);

    for (char c : text) {
        auto it = char_to_id_.find(c);
        ids.push_back(it != char_to_id_.end() ? it->second : UNK_ID);
    }

    if (add_eos) ids.push_back(EOS_ID);
    return ids;
}

// =============================================================================
// decode — token IDs → text (skips special tokens)
// =============================================================================
std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    result.reserve(ids.size());
    for (int id : ids) {
        if (id == PAD_ID || id == BOS_ID || id == EOS_ID) continue;
        if (id == UNK_ID) { result += '?'; continue; }
        if (id >= 0 && id < static_cast<int>(id_to_char_.size()))
            result += id_to_char_[id];
    }
    return result;
}

// =============================================================================
// Lookup helpers
// =============================================================================
int Tokenizer::char_to_id(char c) const {
    auto it = char_to_id_.find(c);
    return it != char_to_id_.end() ? it->second : UNK_ID;
}

std::string Tokenizer::id_to_str(int id) const {
    if (id < 0 || id >= static_cast<int>(id_to_char_.size()))
        return "<INVALID>";
    return id_to_char_[id];
}

bool Tokenizer::has_char(char c) const {
    return char_to_id_.count(c) > 0;
}

// =============================================================================
// print_vocab — dump the full vocabulary
// =============================================================================
void Tokenizer::print_vocab() const {
    std::cout << "Vocabulary (" << vocab_size() << " tokens):\n";
    for (int i = 0; i < vocab_size(); ++i) {
        std::cout << "  [" << i << "] ";
        const std::string& s = id_to_char_[i];
        if (s == "\n")      std::cout << "\\n";
        else if (s == "\t") std::cout << "\\t";
        else if (s == " ")  std::cout << "<SPACE>";
        else                std::cout << s;
        std::cout << "\n";
    }
}

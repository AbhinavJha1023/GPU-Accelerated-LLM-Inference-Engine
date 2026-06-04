// =============================================================================
// test_kv_cache.cpp — GoogleTest tests for KVCache and Tokenizer
// =============================================================================

#include <gtest/gtest.h>
#include "tensor.hpp"
#include "kv_cache.hpp"
#include "tokenizer.hpp"
#include "attention.hpp"

// =============================================================================
// KVCache tests
// =============================================================================
class KVCacheTest : public ::testing::Test {
protected:
    // Small cache for testing: 2 heads, 4 head_dim, max 8 tokens
    KVCache cache{2, 4, 8};   // kv_dim = 2*4 = 8
};

TEST_F(KVCacheTest, InitialState) {
    EXPECT_EQ(cache.length(), 0);
    EXPECT_TRUE(cache.is_empty());
    EXPECT_FALSE(cache.is_full());
    EXPECT_EQ(cache.kv_dim, 8);
}

TEST_F(KVCacheTest, AppendIncreasesLength) {
    Tensor k(1, 8), v(1, 8);
    cache.append(k, v);
    EXPECT_EQ(cache.length(), 1);
    cache.append(k, v);
    EXPECT_EQ(cache.length(), 2);
}

TEST_F(KVCacheTest, GetKeysShape) {
    Tensor k(1, 8, 1.0f), v(1, 8, 2.0f);
    cache.append(k, v);
    cache.append(k, v);
    cache.append(k, v);

    Tensor keys = cache.get_keys();
    EXPECT_EQ(keys.rows, 3);
    EXPECT_EQ(keys.cols, 8);
}

TEST_F(KVCacheTest, GetValuesShape) {
    Tensor k(1, 8), v(1, 8);
    cache.append(k, v);
    cache.append(k, v);

    Tensor vals = cache.get_values();
    EXPECT_EQ(vals.rows, 2);
    EXPECT_EQ(vals.cols, 8);
}

TEST_F(KVCacheTest, StoredDataCorrect) {
    Tensor k(1, 8), v(1, 8);
    // Fill with distinct values
    for (int i = 0; i < 8; ++i) {
        k.data[i] = static_cast<float>(i + 1);
        v.data[i] = static_cast<float>(i + 10);
    }
    cache.append(k, v);

    Tensor keys = cache.get_keys();
    Tensor vals = cache.get_values();
    for (int c = 0; c < 8; ++c) {
        EXPECT_FLOAT_EQ(keys(0, c), static_cast<float>(c + 1));
        EXPECT_FLOAT_EQ(vals(0, c), static_cast<float>(c + 10));
    }
}

TEST_F(KVCacheTest, MultipleRowsInOrder) {
    for (int t = 0; t < 5; ++t) {
        Tensor k(1, 8, static_cast<float>(t));
        Tensor v(1, 8, static_cast<float>(t * 10));
        cache.append(k, v);
    }
    Tensor keys = cache.get_keys();
    // Row 0 should have all elements == 0, row 1 all == 1, etc.
    for (int t = 0; t < 5; ++t)
        for (int c = 0; c < 8; ++c)
            EXPECT_FLOAT_EQ(keys(t, c), static_cast<float>(t));
}

TEST_F(KVCacheTest, Clear) {
    Tensor k(1, 8), v(1, 8);
    cache.append(k, v);
    cache.append(k, v);
    EXPECT_EQ(cache.length(), 2);
    cache.clear();
    EXPECT_EQ(cache.length(), 0);
    EXPECT_TRUE(cache.is_empty());
}

TEST_F(KVCacheTest, OverflowThrows) {
    Tensor k(1, 8), v(1, 8);
    for (int i = 0; i < 8; ++i) cache.append(k, v);   // fill to max
    EXPECT_TRUE(cache.is_full());
    EXPECT_THROW(cache.append(k, v), std::overflow_error);
}

TEST_F(KVCacheTest, WrongDimensionThrows) {
    Tensor k_bad(1, 4), v(1, 8);   // k has wrong dim (4 != 8)
    EXPECT_THROW(cache.append(k_bad, v), std::invalid_argument);
}

TEST_F(KVCacheTest, EmptyGetKeysReturns0Rows) {
    Tensor keys = cache.get_keys();
    EXPECT_EQ(keys.rows, 0);
    EXPECT_EQ(keys.cols, 8);
}

// =============================================================================
// KVCache with MHA integration
// =============================================================================
TEST(KVCacheIntegration, CachedForwardGrowsCache) {
    MultiHeadAttention mha(16, 2);
    KVCache cache(2, 8, 32);   // head_dim=8, kv_dim=16

    EXPECT_EQ(cache.length(), 0);

    for (int t = 0; t < 5; ++t) {
        Tensor tok(1, 16);
        tok.randomize();
        mha.forward_cached(tok, cache);
        EXPECT_EQ(cache.length(), t + 1);
    }
}

TEST(KVCacheIntegration, OutputShapeCorrect) {
    MultiHeadAttention mha(16, 2);
    KVCache cache(2, 8, 32);

    Tensor tok(1, 16);
    tok.randomize();
    Tensor out = mha.forward_cached(tok, cache);
    EXPECT_EQ(out.rows, 1);
    EXPECT_EQ(out.cols, 16);
}

TEST(KVCacheIntegration, NaNFreeGeneration) {
    MultiHeadAttention mha(32, 4);
    KVCache cache(4, 8, 64);

    for (int t = 0; t < 10; ++t) {
        Tensor tok(1, 32);
        tok.randomize(-0.5f, 0.5f);
        Tensor out = mha.forward_cached(tok, cache);
        for (int i = 0; i < out.size(); ++i)
            EXPECT_FALSE(std::isnan(out.data[i]))
                << "NaN at token " << t << " element " << i;
    }
}

// =============================================================================
// Tokenizer tests
// =============================================================================
TEST(TokenizerTest, InitialSpecialTokens) {
    Tokenizer tok;
    EXPECT_EQ(tok.vocab_size(), 4);   // PAD, UNK, BOS, EOS
    EXPECT_EQ(Tokenizer::PAD_ID, 0);
    EXPECT_EQ(Tokenizer::UNK_ID, 1);
    EXPECT_EQ(Tokenizer::BOS_ID, 2);
    EXPECT_EQ(Tokenizer::EOS_ID, 3);
}

TEST(TokenizerTest, BuildVocab) {
    Tokenizer tok;
    tok.build_vocab("abc");
    EXPECT_EQ(tok.vocab_size(), 7);   // 4 special + a + b + c
    EXPECT_TRUE(tok.has_char('a'));
    EXPECT_TRUE(tok.has_char('b'));
    EXPECT_FALSE(tok.has_char('z'));
}

TEST(TokenizerTest, EncodeDecodeRoundtrip) {
    Tokenizer tok;
    tok.build_vocab("Hello World");
    std::string text = "Hello";
    auto ids = tok.encode(text, /*bos=*/true, /*eos=*/true);
    std::string decoded = tok.decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST(TokenizerTest, EncodeAddsSpecialTokens) {
    Tokenizer tok;
    tok.build_vocab("ab");
    auto ids = tok.encode("ab", true, true);
    EXPECT_EQ(ids.front(), Tokenizer::BOS_ID);
    EXPECT_EQ(ids.back(),  Tokenizer::EOS_ID);
    EXPECT_EQ(ids.size(), 4u);   // BOS + 'a' + 'b' + EOS
}

TEST(TokenizerTest, UnknownCharMapsToUnk) {
    Tokenizer tok;
    tok.build_vocab("abc");
    auto ids = tok.encode("axz", false, false);
    EXPECT_EQ(ids[0], tok.char_to_id('a'));
    EXPECT_EQ(ids[1], Tokenizer::UNK_ID);
    EXPECT_EQ(ids[2], Tokenizer::UNK_ID);
}

TEST(TokenizerTest, DecodeSkipsSpecialTokens) {
    Tokenizer tok;
    std::vector<int> ids = {Tokenizer::BOS_ID, 4, 5, Tokenizer::EOS_ID};
    tok.build_vocab("ab");
    std::string dec = tok.decode(ids);
    // BOS and EOS should not appear in output
    EXPECT_EQ(dec.find("<"), std::string::npos);
}

TEST(TokenizerTest, DuplicateCharNotAddedTwice) {
    Tokenizer tok;
    tok.build_vocab("aaa");
    EXPECT_EQ(tok.vocab_size(), 5);   // 4 special + 'a'
}

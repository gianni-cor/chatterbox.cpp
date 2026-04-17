#include "gpt2_bpe.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <functional>
#include <queue>
#include <regex>
#include <sstream>
#include <unordered_set>

// ---- byte-level encoding (GPT-2 convention) --------------------------------

static std::unordered_map<uint8_t, std::string> build_byte_to_unicode() {
    std::unordered_map<uint8_t, std::string> m;
    auto cpt_to_utf8 = [](uint32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) { s += (char)cp; }
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    };
    for (int c = 0x21; c <= 0x7E; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    for (int c = 0xA1; c <= 0xAC; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    for (int c = 0xAE; c <= 0xFF; ++c) m[(uint8_t)c] = cpt_to_utf8((uint32_t)c);
    int n = 0;
    for (int c = 0; c < 256; ++c) {
        if (m.find((uint8_t)c) == m.end()) {
            m[(uint8_t)c] = cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return m;
}

static const std::unordered_map<uint8_t, std::string> & byte_to_unicode() {
    static auto m = build_byte_to_unicode();
    return m;
}

static std::string bytes_to_unicode_str(const std::string & raw) {
    std::string out;
    auto & b2u = byte_to_unicode();
    for (unsigned char c : raw) out += b2u.at(c);
    return out;
}

// ---- simple JSON helpers (no dependencies) ---------------------------------

static std::string json_unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            if (c == '"') out += '"';
            else if (c == '\\') out += '\\';
            else if (c == '/') out += '/';
            else if (c == 'n') out += '\n';
            else if (c == 'r') out += '\r';
            else if (c == 't') out += '\t';
            else if (c == 'u' && i + 4 < s.size()) {
                uint32_t cp = (uint32_t)std::stoul(s.substr(i+1, 4), nullptr, 16);
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && i + 2 < s.size() && s[i+1] == '\\' && s[i+2] == 'u') {
                    uint32_t lo = (uint32_t)std::stoul(s.substr(i+3, 4), nullptr, 16);
                    i += 6;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                if (cp < 0x80) out += (char)cp;
                else if (cp < 0x800) { out += (char)(0xC0|(cp>>6)); out += (char)(0x80|(cp&0x3F)); }
                else if (cp < 0x10000) { out += (char)(0xE0|(cp>>12)); out += (char)(0x80|((cp>>6)&0x3F)); out += (char)(0x80|(cp&0x3F)); }
                else { out += (char)(0xF0|(cp>>18)); out += (char)(0x80|((cp>>12)&0x3F)); out += (char)(0x80|((cp>>6)&0x3F)); out += (char)(0x80|(cp&0x3F)); }
            } else { out += c; }
        } else {
            out += s[i];
        }
    }
    return out;
}

// ---- load vocab.json -------------------------------------------------------

bool gpt2_bpe::load_vocab_json(const std::string & path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "gpt2_bpe: failed to open %s\n", path.c_str()); return false; }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    size_t max_id = 0;
    size_t pos = 0;
    while (pos < json.size()) {
        pos = json.find('"', pos);
        if (pos == std::string::npos) break;
        size_t key_start = ++pos;
        while (pos < json.size() && json[pos] != '"') { if (json[pos] == '\\') ++pos; ++pos; }
        std::string key = json_unescape(json.substr(key_start, pos - key_start));
        ++pos;
        pos = json.find(':', pos);
        if (pos == std::string::npos) break;
        ++pos;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;
        size_t num_start = pos;
        while (pos < json.size() && ((json[pos] >= '0' && json[pos] <= '9') || json[pos] == '-')) ++pos;
        int32_t id = (int32_t)std::stoi(json.substr(num_start, pos - num_start));
        token_to_id[key] = id;
        if ((size_t)id >= max_id) max_id = (size_t)id + 1;
    }

    id_to_token.resize(max_id);
    for (auto & [tok, id] : token_to_id) {
        if (id >= 0 && (size_t)id < id_to_token.size()) id_to_token[id] = tok;
    }
    return true;
}

// ---- load merges.txt -------------------------------------------------------

bool gpt2_bpe::load_merges_txt(const std::string & path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "gpt2_bpe: failed to open %s\n", path.c_str()); return false; }
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string left = line.substr(0, sp);
        std::string right = line.substr(sp + 1);
        while (!right.empty() && (right.back() == '\r' || right.back() == '\n')) right.pop_back();
        bpe_ranks[left + " " + right] = rank++;
    }
    return true;
}

// ---- load added_tokens.json ------------------------------------------------

bool gpt2_bpe::load_added_tokens_json(const std::string & path) {
    std::ifstream f(path);
    if (!f) return true; // optional
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    size_t pos = 0;
    while (pos < json.size()) {
        pos = json.find('"', pos);
        if (pos == std::string::npos) break;
        size_t key_start = ++pos;
        while (pos < json.size() && json[pos] != '"') { if (json[pos] == '\\') ++pos; ++pos; }
        std::string key = json_unescape(json.substr(key_start, pos - key_start));
        ++pos;
        pos = json.find(':', pos);
        if (pos == std::string::npos) break;
        ++pos;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) ++pos;
        size_t ns = pos;
        while (pos < json.size() && ((json[pos] >= '0' && json[pos] <= '9') || json[pos] == '-')) ++pos;
        int32_t id = (int32_t)std::stoi(json.substr(ns, pos - ns));

        token_to_id[key] = id;
        if ((size_t)id >= id_to_token.size()) id_to_token.resize((size_t)id + 1);
        id_to_token[id] = key;
    }
    return true;
}

// ---- in-memory loader for GGUF-embedded tokenizer -------------------------

bool gpt2_bpe::load_from_arrays(const std::vector<std::string> & tokens,
                                const std::vector<std::string> & merges) {
    if (tokens.empty()) return false;

    id_to_token = tokens;
    token_to_id.clear();
    token_to_id.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        token_to_id[tokens[i]] = (int32_t) i;
    }

    bpe_ranks.clear();
    bpe_ranks.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        bpe_ranks[merges[i]] = (int) i;
    }
    return true;
}

// ---- GPT-2 pre-tokenization regex ------------------------------------------
// Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

static std::vector<std::string> gpt2_regex_split(const std::string & text) {
    static const std::regex re(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)",
        std::regex::optimize);

    std::vector<std::string> words;
    auto begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        words.push_back(it->str());
    }
    return words;
}

// ---- BPE merge algorithm ---------------------------------------------------

static int find_rank(const std::unordered_map<std::string, int> & ranks,
                     const std::string & left, const std::string & right) {
    auto it = ranks.find(left + " " + right);
    return it != ranks.end() ? it->second : -1;
}

static std::vector<std::string> bpe_merge(const std::string & token,
                                          const std::unordered_map<std::string, int> & ranks) {
    // split token into individual UTF-8 characters
    std::vector<std::string> parts;
    for (size_t i = 0; i < token.size(); ) {
        size_t len = 1;
        unsigned char c = (unsigned char)token[i];
        if      ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        parts.push_back(token.substr(i, len));
        i += len;
    }

    while (parts.size() >= 2) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            int r = find_rank(ranks, parts[i], parts[i+1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
        }
        if (best_rank == INT32_MAX) break;
        parts[best_i] = parts[best_i] + parts[best_i + 1];
        parts.erase(parts.begin() + (int)best_i + 1);
    }
    return parts;
}

// ---- tokenize --------------------------------------------------------------

std::vector<int32_t> gpt2_bpe::tokenize(const std::string & text) const {
    std::vector<int32_t> ids;
    if (text.empty()) return ids;

    // check added tokens first (paralinguistic tags like [laugh])
    struct added_span { size_t start; size_t len; int32_t id; };
    std::vector<added_span> spans;
    for (auto & [tok, id] : token_to_id) {
        if (id < 50257) continue; // only added tokens
        size_t pos = 0;
        while ((pos = text.find(tok, pos)) != std::string::npos) {
            spans.push_back({pos, tok.size(), id});
            pos += tok.size();
        }
    }
    std::sort(spans.begin(), spans.end(), [](const added_span & a, const added_span & b) { return a.start < b.start; });

    // remove overlapping spans (keep earliest)
    std::vector<added_span> clean;
    size_t last_end = 0;
    for (auto & sp : spans) {
        if (sp.start >= last_end) { clean.push_back(sp); last_end = sp.start + sp.len; }
    }

    auto tokenize_fragment = [&](const std::string & frag) {
        auto words = gpt2_regex_split(frag);
        for (auto & word : words) {
            std::string bpe_input = bytes_to_unicode_str(word);
            auto parts = bpe_merge(bpe_input, bpe_ranks);
            for (auto & part : parts) {
                auto it = token_to_id.find(part);
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                } else {
                    for (unsigned char c : word) {
                        auto & b2u = byte_to_unicode();
                        auto jt = token_to_id.find(b2u.at(c));
                        if (jt != token_to_id.end()) ids.push_back(jt->second);
                    }
                }
            }
        }
    };

    size_t cursor = 0;
    for (auto & sp : clean) {
        if (sp.start > cursor) tokenize_fragment(text.substr(cursor, sp.start - cursor));
        ids.push_back(sp.id);
        cursor = sp.start + sp.len;
    }
    if (cursor < text.size()) tokenize_fragment(text.substr(cursor));

    return ids;
}

// ---- punc_norm (matches Python tts_turbo.py) -------------------------------

std::string gpt2_bpe::punc_norm(const std::string & text) {
    if (text.empty()) return "You need to add some text for me to talk.";

    std::string t = text;

    // capitalize first letter
    if (t[0] >= 'a' && t[0] <= 'z') t[0] = t[0] - 'a' + 'A';

    // collapse multiple spaces
    {
        std::string r;
        bool prev_space = false;
        for (char c : t) {
            if (c == ' ') { if (!prev_space) r += c; prev_space = true; }
            else { r += c; prev_space = false; }
        }
        t = r;
    }

    // replace uncommon punctuation
    auto replace_all = [](std::string & s, const std::string & from, const std::string & to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(t, "\xe2\x80\xa6", ", ");   // …
    replace_all(t, ":", ",");
    replace_all(t, "\xe2\x80\x94", "-");     // —
    replace_all(t, "\xe2\x80\x93", "-");     // –
    replace_all(t, " ,", ",");
    replace_all(t, "\xe2\x80\x9c", "\"");    // "
    replace_all(t, "\xe2\x80\x9d", "\"");    // "
    replace_all(t, "\xe2\x80\x98", "'");     // '
    replace_all(t, "\xe2\x80\x99", "'");     // '

    // strip trailing spaces
    while (!t.empty() && t.back() == ' ') t.pop_back();

    // add period if no ending punctuation
    if (!t.empty()) {
        char last = t.back();
        if (last != '.' && last != '!' && last != '?' && last != '-' && last != ',')
            t += '.';
    }

    return t;
}

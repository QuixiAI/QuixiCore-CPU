#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace qcb {

// Minimal append-only JSON writer. Emission order == call order, so field
// order in the output is stable. Handles only what this harness produces
// (ASCII content); no third-party JSON dependency.
class JsonWriter {
 public:
  const std::string& str() const { return out_; }

  void begin_obj() {
    value_prefix();
    out_ += '{';
    first_.push_back(true);
  }
  void end_obj() {
    out_ += '}';
    first_.pop_back();
  }
  void begin_arr() {
    value_prefix();
    out_ += '[';
    first_.push_back(true);
  }
  void end_arr() {
    out_ += ']';
    first_.pop_back();
  }

  void key(std::string_view k) {
    if (!first_.empty() && !first_.back()) {
      out_ += ", ";
    }
    if (!first_.empty()) {
      first_.back() = false;
    }
    append_string(k);
    out_ += ": ";
    have_key_ = true;
  }

  void val(std::string_view s) {
    value_prefix();
    append_string(s);
  }
  void val(const char* s) { val(std::string_view(s)); }
  void val(bool b) {
    value_prefix();
    out_ += b ? "true" : "false";
  }
  void val(long long i) {
    value_prefix();
    out_ += std::to_string(i);
  }
  void val(int i) { val(static_cast<long long>(i)); }
  void val(double d) {
    value_prefix();
    if (!std::isfinite(d)) {
      out_ += "null";  // JSON has no NaN/Inf (e.g. cv when mean is zero).
      return;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.12g", d);
    out_ += buf;
  }
  void null() {
    value_prefix();
    out_ += "null";
  }

 private:
  void value_prefix() {
    if (have_key_) {
      have_key_ = false;
      return;
    }
    if (!first_.empty()) {
      if (!first_.back()) {
        out_ += ", ";
      }
      first_.back() = false;
    }
  }

  void append_string(std::string_view s) {
    out_ += '"';
    for (const char ch : s) {
      const unsigned char c = static_cast<unsigned char>(ch);
      if (ch == '"' || ch == '\\') {
        out_ += '\\';
        out_ += ch;
      } else if (c >= 0x20) {
        out_ += ch;
      } else if (ch == '\n') {
        out_ += "\\n";
      } else if (ch == '\t') {
        out_ += "\\t";
      } else if (ch == '\r') {
        out_ += "\\r";
      } else {
        char buf[8];
        std::snprintf(buf, sizeof buf, "\\u%04x", c);
        out_ += buf;
      }
    }
    out_ += '"';
  }

  std::string out_;
  std::vector<bool> first_;
  bool have_key_ = false;
};

}  // namespace qcb


#include <iostream>
#include <iomanip>
#include <fstream>

#include "../src/fasttext_api.h"

static std::vector<std::string> load_subset(const std::string &subset_file) {
  std::vector<std::string> subset;
  std::ifstream ifs_subset_file(subset_file);

  if (!ifs_subset_file.is_open()) {
    std::cout << "subset file " << subset_file << " open failed." << std::endl;
    abort();
  }

  std::string line;
  while (!ifs_subset_file.eof()) {
    std::getline(ifs_subset_file, line);
    if (line.empty()) {
      continue;
    }
    subset.push_back(line);
  }

  ifs_subset_file.close();
  return subset;
}

static bool find_item(const std::vector<std::pair<float, std::string>> &predictions,
    const std::string &item) {
  for (auto &p : predictions) {
    if (item == p.second) {
      return true;
    }
  }
  return false;
}

int main(int argc, char *argv[]) {
  fasttext::FastTextApi fapi;
  if (argc != 6) {
    std::cout
      << "Usage: lsh_benchmark <model> <lsh-build-tree> <lsh-dict> <subset> <k>"
      << std::endl;
    exit(-1);
  }

  fapi.LoadModel(argv[1]);
  fapi.PrecomputeWordVectors();
  auto subset = load_subset(argv[4]);
  fapi.FeedSubset(subset);
  fapi.LoadLSH(argv[2], argv[3]);

  std::cout << "Load done." << std::endl;

  int k = std::stoi(argv[5]);
  std::vector<std::string> inputs;
  size_t total = 0;
  size_t valid = 0;
  size_t processed = 0;

  std::cout << std::setprecision(3);
  for (auto &s : subset) {
    inputs.clear();
    inputs.push_back(s);
    auto predictions = fapi.XcbowPredictSubset(inputs, k);
    auto predictions_lsh = fapi.XcbowPredictLSH(inputs, k);

    total += predictions.size();
    for (auto &p : predictions_lsh) {
      const std::string &item = p.second;
      if (find_item(predictions, item)) {
        ++valid;
      }
    }
    ++processed;
    if (processed % 100 == 0) {
      std::cout << processed * 1.0 / subset.size() * 100 << "% processed, P = "
        << valid * 1.0 / total * 100 << "%"
        << ", valid = " << valid
        <<", total = " << total
        << std::endl;
    }
  }

  std::cout << "total: " << total << std::endl;
  std::cout << "valid: " << valid << std::endl;
  std::cout << "final P = " << valid * 1.0 / total * 100 << "%" << std::endl;

  return 0;
}

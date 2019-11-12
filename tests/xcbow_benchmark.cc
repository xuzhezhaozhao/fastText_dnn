
#include <sys/time.h>

#include <fstream>
#include <iomanip>
#include <iostream>

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

int main(int argc, char *argv[]) {
  fasttext::FastTextApi fapi;
  if (argc != 4) {
    std::cout << "Usage: xcbow_benchmark <model> <subset> <k>" << std::endl;
    exit(-1);
  }

  fapi.LoadModel(argv[1]);
  fapi.PrecomputeWordVectors();
  auto subset = load_subset(argv[2]);
  fapi.FeedSubset(subset);
  std::cout << "Load done." << std::endl;

  int k = std::stoi(argv[3]);
  std::vector<std::string> inputs;
  size_t processed = 0;
  long long total_time = 0;

  struct timeval stv;
  gettimeofday(&stv, NULL);
  long long start = stv.tv_sec * 1000 + stv.tv_usec / 1000;

  std::cout << std::setprecision(3);
  for (auto &s : subset) {
    inputs.clear();
    inputs.push_back(s);
    auto predictions = fapi.XcbowPredictSubset(inputs, k);
    gettimeofday(&stv, NULL);
    long long end = stv.tv_sec * 1000 + stv.tv_usec / 1000;
    total_time = end - start;
    ++processed;
    if (processed % 100 == 0) {
      std::cout << "processed " << processed
                << ", time avg: " << (double)total_time / processed << " ms"
                << std::endl;
    }
  }

  return 0;
}

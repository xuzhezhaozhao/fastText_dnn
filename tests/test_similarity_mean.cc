
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/fasttext_api.h"

namespace {
  void printUsage() {
    std::cerr
        << "Usage: ./test <model> <ntarget> <nquery> <target ...> <query ...>"
        << std::endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 6) {
    printUsage();
    exit(1);
  }
  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  fapi.PrecomputeWordVectors(true);

  int ntarget = std::stoi(argv[2]);
  int nquery = std::stoi(argv[3]);

  if (argc < 4 + ntarget + nquery) {
    printUsage();
    exit(1);
  }

  std::vector<std::string> targets;
  for (int i = 0; i < ntarget; ++i) {
    targets.push_back(argv[4 + i]);
  }

  std::vector<std::string> queries;
  for (int i = 0; i < nquery; ++i) {
    queries.push_back(argv[4 + ntarget + i]);
  }

  std::vector<std::vector<std::string>> Q_list;
  Q_list.push_back(queries);

  auto similarity = fapi.ComputeSimilarityMean(targets, Q_list);
  for (auto f : similarity) {
    std::cout << f << std::endl;
  }

  return 0;
}

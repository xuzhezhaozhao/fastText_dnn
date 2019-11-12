
#include <iostream>
#include <vector>
#include <utility>
#include <string>

#include "../src/fasttext_api.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: ./test <model> <target> <query> ..." << std::endl;
    exit(1);
  }
  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  fapi.PrecomputeWordVectors(true);
  std::string target = argv[2];
  std::vector<std::string> queries;
  for (int i = 3; i < argc; ++i) {
    queries.push_back(argv[i]);
  }

  auto similarity = fapi.ComputeSimilarity(target, queries);
  for (auto f : similarity) {
    std::cout << f << std::endl;
  }

  return 0;
}

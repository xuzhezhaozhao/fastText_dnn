
#include <iostream>

#include "../src/fasttext_api.h"

int main(int argc, char *argv[]) {
  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  fapi.PrecomputeWordVectors();

  std::vector<std::string> inputs;
  for (int i = 2; i < argc; ++i) {
    inputs.push_back(argv[i]);
  }
  auto preditions = fapi.NNWithVector(inputs, 10);

  for (auto it = preditions.cbegin(); it != preditions.cend(); ++it) {
    std::cout << it->second << " : " << it->first << std::endl;
  }

  return 0;
}

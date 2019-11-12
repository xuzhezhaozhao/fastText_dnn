#include <iostream>
#include <vector>
#include <utility>
#include <string>

#include "../src/fasttext_api.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: <model> <query> <k>" << std::endl;
    exit(-1);
  }
  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  fapi.PrecomputeLabelsMatrix();
  auto nn = fapi.NNLabels(argv[2], std::stoi(argv[3]));
  for (auto it = nn.cbegin(); it != nn.cend(); ++it) {
    std::cout << it->second << " : " << it->first << std::endl;
  }
}

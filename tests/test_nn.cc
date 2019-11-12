
#include <iostream>
#include <vector>
#include <utility>
#include <string>

#include "../src/fasttext_api.h"

int main(int argc, char *argv[]) {
  (void)argc;
  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  fapi.PrecomputeWordVectors();
  std::vector<std::string> watched;
  for (int i = 2; i < argc; ++i) {
    watched.push_back(argv[i]);
  }

  auto nn = fapi.PCTRPredict(watched, 10);
  for (auto it = nn.cbegin(); it != nn.cend(); ++it) {
    std::cout << it->second << " : " << it->first << std::endl;
  }
}

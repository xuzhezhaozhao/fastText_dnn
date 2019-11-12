#include <iostream>
#include <vector>
#include <utility>
#include <string>

#include "../src/fasttext_api.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: <model> <k> <filename>" << std::endl;
    exit(-1);
  }

  fasttext::FastTextApi fapi;
  fapi.LoadModel(argv[1]);
  fapi.PrecomputeLabelsMatrix();
  int k = std::stoi(argv[2]);
  const auto &dict = fapi.fasttext().getDictionary();
  auto args = fapi.fasttext().getArgs();
  std::ofstream ofs(argv[3]);
  if (!ofs.is_open()) {
    std::cout << "open '" << argv[3] << "'" << " failed." << std::endl;
    exit(-1);
  }

  const int TMP_SIZE = 64;
  char tmp[TMP_SIZE];
  for (int i = 0; i < dict->nlabels(); ++i) {
    auto label = dict->getLabel(i).substr(args.label.size());
    ofs.write(label.data(), label.size());
    ofs.write("\t", 1);
    auto nn = fapi.NNLabels(label, k);
    for (auto& p : nn) {
      ofs.write(p.second.data(), p.second.size());
      ofs.write(":", 1);
      auto to_write = std::to_string(p.first);
      snprintf(tmp, TMP_SIZE, "%.1f", p.first);
      ofs.write(tmp, strlen(tmp));
      ofs.write(" ", 1);
    }
    ofs.write("\n", 1);
  }
}

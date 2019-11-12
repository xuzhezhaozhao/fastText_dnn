
#include <fstream>
#include <iostream>
#include <map>

#include "../src/annoylib.h"
#include "../src/kissrandom.h"

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("Usage: lsh_query <build-tree-file> <dict-file> <dim> <k>\n\n");
    exit(-1);
  }

  std::cout << "load dict ..." << std::endl;
  std::ifstream dict_file(argv[2]);
  if (!dict_file.is_open()) {
    std::cout << "open dict file failed." << std::endl;
    exit(-1);
  }
  std::vector<std::string> idx2rowkey;
  std::map<std::string, size_t> rowkey2idx;
  std::string rowkey;
  while (!dict_file.eof()) {
    std::getline(dict_file, rowkey);
    if (rowkey.empty()) {
      continue;
    }
    rowkey2idx[rowkey] = idx2rowkey.size();
    idx2rowkey.push_back(rowkey);
  }

  int dim = std::stoi(argv[3]);
  AnnoyIndex<int, float, Euclidean, Kiss32Random> annoy_index =
      AnnoyIndex<int, float, Euclidean, Kiss32Random>(dim);

  annoy_index.load(argv[1]);

  int k = std::stoi(argv[4]);
  std::string query;
  std::cout << "query word: ";
  while (std::cin >> query) {
    auto it = rowkey2idx.find(query);
    if (it == rowkey2idx.end()) {
      std::cout << "not exists" << std::endl;
      continue;
    }

    size_t idx = rowkey2idx[query];
    std::vector<int> closest;
    std::vector<float> dist;
    size_t n = rowkey2idx.size();
    annoy_index.get_nns_by_item(idx, k, n, &closest, &dist);
    for (size_t i = 0; i < closest.size(); ++i) {
      std::cout << idx2rowkey[closest[i]] << ": " << dist[i] << std::endl;
    }

    std::cout << "query word: ";
  }

  return 0;
}


#include <fstream>
#include <iostream>
#include <map>

#include "../src/annoylib.h"
#include "../src/kissrandom.h"

static void normalize(std::vector<float> &v) {
  float sum = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    sum += v[i] * v[i];
  }

  if (sum == 0) {
    return;
  }

  for (auto &x : v) {
    x /= sum;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: lsh_build_tree <vector-file> <build-tree-file> <q>\n\n");
    exit(-1);
  }
  std::ifstream in(argv[1]);
  if (!in.is_open()) {
    std::cout << "open vector file failed." << std::endl;
    exit(-1);
  }

  int n, dim;
  in >> n >> dim;
  std::cout << n << " : " << dim << std::endl;

  AnnoyIndex<int, float, Euclidean, Kiss32Random> annoy_index =
      AnnoyIndex<int, float, Euclidean, Kiss32Random>(dim);

  std::cout << "load vector ..." << std::endl;
  for (int i = 0; i < n; ++i) {
    std::vector<float> v;
    float d;
    std::string rowkey;
    in >> rowkey;
    for (int j = 0; j < dim; ++j) {
      in >> d;
      v.push_back(d);
    }
    //normalize(v);
    annoy_index.add_item(i, v);
  }
  std::cout << "load vector done" << std::endl;

  std::cout << "build ..." << std::endl;
  int q = std::stoi(argv[3]);
  annoy_index.build(q);
  std::cout << "build done" << std::endl;

  annoy_index.save(argv[2]);

  return 0;
}

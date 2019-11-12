/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>
#include <sstream>

#include "fasttext.h"
#include "args.h"
#include "mockmain.h"

using namespace fasttext;

void printUsage() {
  std::cerr
    << "usage: fasttext <command> <args>\n\n"
    << "The commands supported by fasttext are:\n\n"
    << "  supervised              train a supervised classifier\n"
    << "  quantize                quantize a model to reduce the memory usage\n"
    << "  test                    evaluate a supervised classifier\n"
    << "  pctr-test               evaluate a pctr classifier\n"
    << "  predict                 predict most likely labels\n"
    << "  predict-prob            predict most likely labels with probabilities\n"
    << "  pctr-predict            predict most likely pctr labels\n"
    << "  xcbow-predict           xcbow predict most likely pctr labels\n"
    << "  xcbow-predict-lsh       xcbow predict lsh most likely pctr labels\n"
    << "  skipgram                train a skipgram model\n"
    << "  cbow                    train a cbow model\n"
    << "  xcbow                   train a xcbow model\n"
    << "  print-word-vectors      print word vectors given a trained model\n"
    << "  print-sentence-vectors  print sentence vectors given a trained model\n"
    << "  nn                      query for nearest neighbors\n"
    << "  nnsubset                query for nearest neighbors in a subset\n"
    << "  multi-nn                query for nearest neighbors using multi threads\n"
    << "  multi-nnsubset          query for nearest neighbors in a subset using multi threads\n"
    << "  analogies               query for analogies\n"
    << std::endl;
}

void printQuantizeUsage() {
  std::cerr
    << "usage: fasttext quantize <args>"
    << std::endl;
}

void printTestUsage() {
  std::cerr
    << "usage: fasttext test <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPctrTestUsage() {
  std::cerr
    << "usage: fasttext pctr-test <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPredictUsage() {
  std::cerr
    << "usage: fasttext predict[-prob] <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPctrPredictUsage() {
  std::cerr
    << "usage: fasttext pctr-predict <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printXcbowPredictUsage() {
  std::cerr
    << "usage: fasttext xcbow-predict <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printXcbowPredictLSHUsage() {
  std::cerr
    << "usage: fasttext xcbow-predict-lsh <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPrintWordVectorsUsage() {
  std::cerr
    << "usage: fasttext print-word-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void printPrintSentenceVectorsUsage() {
  std::cerr
    << "usage: fasttext print-sentence-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void printPrintNgramsUsage() {
  std::cerr
    << "usage: fasttext print-ngrams <model> <word>\n\n"
    << "  <model>      model filename\n"
    << "  <word>       word to print\n"
    << std::endl;
}

void quantize(const std::vector<std::string>& args) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  if (args.size() < 3) {
    printQuantizeUsage();
    a->printHelp();
    exit(EXIT_FAILURE);
  }
  a->parseArgs(args);
  FastText fasttext;
  // parseArgs checks if a->output is given.
  fasttext.loadModel(a->output + ".bin");
  fasttext.quantize(a);
  fasttext.saveModel();
  exit(0);
}

void printNNUsage() {
  std::cout
    << "usage: fasttext nn <model> <k> <t>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k words\n"
    << "  <t>          (optional; 0 by default) only occurance count larger than t will be predicted"
    << std::endl;
}

void printNNLabelsUsage() {
  std::cout
    << "usage: fasttext nn <model> <k> <t>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << "  <t>          (optional; 0 by default) only occurance count larger than t will be predicted"
    << std::endl;
}

void printNNLSHUsage() {
  std::cout
    << "usage: fasttext nnlsh <model> <k> <t>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << "  <t>          (optional; 0 by default) only occurance count larger than t will be predicted"
    << std::endl;
}
void printNNSubsetUsage() {
  std::cout
    << "usage: fasttext nnsubset <model> <subset> <k> <t>\n\n"
    << "  <model>      model filename\n"
    << "  <subset>     subset filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << "  <t>          (optional; 0 by default) only occurance count larger than t will be predicted"
    << std::endl;
}

void printAnalogiesUsage() {
  std::cout
    << "usage: fasttext analogies <model> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
}

void printMultiNNUsage() {
  std::cout
    << "usage: fasttext multi-nn <model> <query-file> <thread> <k>\n\n"
    << std::endl;
}
void printMultiNNSubsetUsage() {
  std::cout
    << "usage: fasttext multi-nnsubset <model> <query-file> <subset> <thread> <k>\n\n"
    << std::endl;
}

void test(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 5) {
    printTestUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (args.size() >= 5) {
    k = std::stoi(args[4]);
  }

  FastText fasttext;
  fasttext.loadModel(args[2]);

  std::string infile = args[3];
  if (infile == "-") {
    fasttext.test(std::cin, k);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.test(ifs, k);
    ifs.close();
  }
  exit(0);
}

void pctr_test(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 5) {
    printPctrTestUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (args.size() >= 5) {
    k = std::stoi(args[4]);
  }

  FastText fasttext;
  fasttext.loadModel(args[2]);

  std::string infile = args[3];
  if (infile == "-") {
    fasttext.pctrTest(std::cin, k);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.pctrTest(ifs, k);
    ifs.close();
  }
  exit(0);
}

void predict(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 5) {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (args.size() >= 5) {
    k = std::stoi(args[4]);
  }

  bool print_prob = args[1] == "predict-prob";
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string infile(args[3]);
  if (infile == "-") {
    fasttext.predict(std::cin, k, print_prob);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.predict(ifs, k, print_prob);
    ifs.close();
  }

  exit(0);
}

void pctr_predict(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 5) {
    printPctrPredictUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (args.size() >= 5) {
    k = std::stoi(args[4]);
  }

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string infile(args[3]);
  if (infile == "-") {
    fasttext.pctrPredict(std::cin, k);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.pctrPredict(ifs, k);
    ifs.close();
  }

  exit(0);
}

void xcbow_predict(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 5) {
    printXcbowPredictUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (args.size() >= 5) {
    k = std::stoi(args[4]);
  }

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string infile(args[3]);
  if (infile == "-") {
    // TODO
    // fasttext.XcbowPredict(std::cin, k);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    // TODO
    // fasttext.XcbowPredict(ifs, k);
    ifs.close();
  }

  exit(0);
}

void xcbow_predict_lsh(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 5) {
    printPctrPredictUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (args.size() >= 5) {
    k = std::stoi(args[4]);
  }

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string infile(args[3]);
  if (infile == "-") {
    // TODO
    // fasttext.XcbowPredictLSH(std::cin, k);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    // TODO
    // fasttext.XcbowPredictLSH(ifs, k);
    ifs.close();
  }

  exit(0);
}

void printWordVectors(const std::vector<std::string> args) {
  if (args.size() != 3) {
    printPrintWordVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string word;
  Vector vec(fasttext.getDimension());
  while (std::cin >> word) {
    fasttext.getWordVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
  exit(0);
}

void printSentenceVectors(const std::vector<std::string> args) {
  if (args.size() != 3) {
    printPrintSentenceVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  Vector svec(fasttext.getDimension());
  while (std::cin.peek() != EOF) {
    fasttext.getSentenceVector(std::cin, svec);
    // Don't print sentence
    std::cout << svec << std::endl;
  }
  exit(0);
}

void printNgrams(const std::vector<std::string> args) {
  if (args.size() != 4) {
    printPrintNgramsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.ngramVectors(std::string(args[3]));
  exit(0);
}

void nn(const std::vector<std::string> args) {
  int32_t k, t;
  if (args.size() == 3) {
    k = 10;
    t = 0;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
    t = 0;
  } else if (args.size() == 5) {
    k = std::stoi(args[3]);
    t = std::stoi(args[4]);
  } else {
    printNNUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.nn(k, t);
  exit(0);
}

void nnlabels(const std::vector<std::string> args) {
  int32_t k, t;
  if (args.size() == 3) {
    k = 10;
    t = 0;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
    t = 0;
  } else if (args.size() == 5) {
    k = std::stoi(args[3]);
    t = std::stoi(args[4]);
  } else {
    printNNLabelsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.nnlabels(k, t);
  exit(0);
}

void nnlsh(const std::vector<std::string> args) {
  int32_t k, t;
  if (args.size() == 3) {
    k = 10;
    t = 0;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
    t = 0;
  } else if (args.size() == 5) {
    k = std::stoi(args[3]);
    t = std::stoi(args[4]);
  } else {
    printNNLSHUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.nnlsh(k, t);
  exit(0);
}

static void readsubset(const std::string &subsetfile,
    std::vector<std::string> &subset) {
  std::ifstream ifs_subset(subsetfile);
  if (!ifs_subset.is_open()) {
    std::cerr << "subset file: " << subsetfile << " open failed."  << std::endl;
    exit(-1);
  }
  std::string line;
  while (!ifs_subset.eof()) {
    std::getline(ifs_subset, line);
    if (line.empty()) {
      continue;
    }
    subset.push_back(line);
  }
}

void nnsubset(const std::vector<std::string> &args) {
  int32_t k, t;
  if (args.size() == 4) {
    k = 10;
    t = 0;
  } else if (args.size() == 5) {
    k = std::stoi(args[4]);
    t = 0;
  } else if (args.size() == 6) {
    k = std::stoi(args[4]);
    t = std::stoi(args[5]);
  } else {
    printNNSubsetUsage();
    exit(EXIT_FAILURE);
  }

  std::vector<std::string> subset;
  readsubset(args[3], subset);

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.feedsubset(subset);
  fasttext.nnsubset(k, t);
  exit(0);
}

void multinn(const std::vector<std::string> &args) {
  // fasttext multi-nn <model> <query-file> <thread> <k>
  if (args.size() < 6) {
    printMultiNNUsage();
    exit(-1);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  int threads = std::stoi(args[4]);
  int k = std::stoi(args[5]);
  fasttext.multinn(args[3], threads, k, false);

  exit(0);
}

void multinnsubset(const std::vector<std::string> &args) {
  // fasttext multi-nn <model> <query-file> <subset> <thread> <k>
  if (args.size() < 7) {
    printMultiNNSubsetUsage();
    exit(-1);
  }

  std::vector<std::string> subset;
  readsubset(args[4], subset);

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.feedsubset(subset);
  int threads = std::stoi(args[5]);
  int k = std::stoi(args[6]);
  fasttext.multinn(args[3], threads, k, true);

  exit(0);
}

void analogies(const std::vector<std::string> args) {
  int32_t k;
  if (args.size() == 3) {
    k = 10;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
  } else {
    printAnalogiesUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.analogies(k);
  exit(0);
}

void train(const std::vector<std::string> args) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  a->parseArgs(args);
  FastText fasttext;
  fasttext.train(a);
  fasttext.saveModel();
  fasttext.saveVectors();
  if (a->saveOutput > 0) {
    fasttext.saveOutput();
  }
}

int fasttext::mockmain(const std::vector<std::string> &args) {
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(args[1]);
  if (command == "skipgram" || command == "cbow" || command == "supervised"
      || command == "xcbow" || command == "pctr") {
    train(args);
  } else if (command == "test") {
    test(args);
  } else if (command == "pctr-test") {
    pctr_test(args);
  } else if (command == "quantize") {
    quantize(args);
  } else if (command == "print-word-vectors") {
    printWordVectors(args);
  } else if (command == "print-sentence-vectors") {
    printSentenceVectors(args);
  } else if (command == "print-ngrams") {
    printNgrams(args);
  } else if (command == "nn") {
    nn(args);
  } else if (command == "nnlabels") {
    nnlabels(args);
  } else if (command == "nnlsh") {
    nnlsh(args);
  } else if (command == "nnsubset") {
    nnsubset(args);
  } else if (command == "multi-nn") {
    multinn(args);
  } else if (command == "multi-nnsubset") {
    multinnsubset(args);
  } else if (command == "analogies") {
    analogies(args);
  } else if (command == "predict" || command == "predict-prob" ) {
    predict(args);
  } else if (command == "pctr-predict") {
    pctr_predict(args);
  } else if (command == "xcbow-predict") {
    xcbow_predict(args);
  } else if (command == "xcbow-predict-lsh") {
    xcbow_predict_lsh(args);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}

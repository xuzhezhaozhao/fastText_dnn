/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"

#include <assert.h>
#include <math.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "deps/openblas/include/cblas.h"

namespace fasttext {

FastText::FastText() : quant_(false) {}

void FastText::addInputVector(Vector& vec, int32_t ind) const {
  if (quant_) {
    vec.addRow(*qinput_, ind);
  } else {
    vec.addRow(*input_, ind);
  }
}

void FastText::addLabelVector(Vector& vec, int32_t ind) const {
  assert(ind >= 0);
  assert(ind < output_->m_);
  vec.addRow(*output_, ind);
}

std::shared_ptr<const Dictionary> FastText::getDictionary() const {
  return dict_;
}

const Args FastText::getArgs() const { return *args_.get(); }

std::shared_ptr<const Matrix> FastText::getInputMatrix() const {
  return input_;
}

std::shared_ptr<const Matrix> FastText::getOutputMatrix() const {
  return output_;
}

int32_t FastText::getWordId(const std::string& word) const {
  return dict_->getId(word);
}

int32_t FastText::getSubwordId(const std::string& word) const {
  int32_t h = dict_->hash(word) % args_->bucket;
  return dict_->nwords() + h;
}

void FastText::getWordVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
  vec.zero();
  for (int i = 0; i < ngrams.size(); i++) {
    addInputVector(vec, ngrams[i]);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void FastText::getLabelVector(Vector& vec, const std::string& label) const {
  vec.zero();
  int32_t id = dict_->getId(label);
  if (id < 0) {
    return;
  }
  int32_t lid = id - dict_->nwords();
  assert(lid >= 0);
  addLabelVector(vec, lid);
}

void FastText::getVector(Vector& vec, const std::string& word) const {
  getWordVector(vec, word);
}

void FastText::getSubwordVector(Vector& vec, const std::string& subword) const {
  vec.zero();
  int32_t h = dict_->hash(subword) % args_->bucket;
  h = h + dict_->nwords();
  addInputVector(vec, h);
}

void FastText::saveVectors() {
  std::ofstream ofs(args_->output + ".vec");
  if (!ofs.is_open()) {
    std::cerr << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    const std::string& word = dict_->getWord(i);
    getWordVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveOutput() {
  std::ofstream ofs(args_->output + ".output");
  if (!ofs.is_open()) {
    std::cerr << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (quant_) {
    std::cerr << "Option -saveOutput is not supported for quantized models."
              << std::endl;
    return;
  }
  int32_t n =
      (args_->model == model_name::sup || args_->model == model_name::pctr)
          ? dict_->nlabels()
          : dict_->nwords();
  ofs << n << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < n; i++) {
    std::string word =
        (args_->model == model_name::sup || args_->model == model_name::pctr)
            ? dict_->getLabel(i)
            : dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

bool FastText::checkModel(std::istream& in) {
  int32_t magic;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version > FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

void FastText::signModel(std::ostream& out) {
  const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
  const int32_t version = FASTTEXT_VERSION;
  out.write((char*)&(magic), sizeof(int32_t));
  out.write((char*)&(version), sizeof(int32_t));
}

void FastText::saveModel() {
  std::string fn(args_->output);
  if (quant_) {
    fn += ".ftz";
  } else {
    fn += ".bin";
  }
  saveModel(fn);
}

void FastText::saveModel(const std::string path) {
  std::ofstream ofs(path, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw std::invalid_argument(path + " cannot be opened for saving!");
  }
  signModel(ofs);
  args_->save(ofs);
  dict_->save(ofs);

  ofs.write((char*)&(quant_), sizeof(bool));
  if (quant_) {
    qinput_->save(ofs);
  } else {
    input_->save(ofs);
  }

  ofs.write((char*)&(args_->qout), sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->save(ofs);
  } else {
    output_->save(ofs);
  }

  ofs.close();
}

void FastText::loadModel(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
}

void FastText::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  qinput_ = std::make_shared<QMatrix>();
  qoutput_ = std::make_shared<QMatrix>();
  args_->load(in);
  if (version == 11 && args_->model == model_name::sup) {
    // backward compatibility: old supervised models do not use char ngrams.
    args_->maxn = 0;
  }
  dict_->load(in);

  bool quant_input;
  in.read((char*)&quant_input, sizeof(bool));
  if (quant_input) {
    quant_ = true;
    qinput_->load(in);
  } else {
    input_->load(in);
  }

  if (!quant_input && dict_->isPruned()) {
    std::cerr << "Invalid model file.\n"
              << "Please download the updated model from www.fasttext.cc.\n"
              << "See issue #332 on Github for more information.\n";
    exit(1);
  }

  in.read((char*)&args_->qout, sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->load(in);
  } else {
    output_->load(in);
  }

  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);

  if (args_->model == model_name::sup || args_->model == model_name::pctr) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

void FastText::printInfo(real progress, real loss) {
  real t = real(clock() - start) / CLOCKS_PER_SEC;
  real wst = real(tokenCount) / t;
  real lr = args_->lr * (1.0 - progress);
  int eta = int(t / progress * (1 - progress) / args_->thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;
  std::cerr << std::fixed;
  std::cerr << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::cerr << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::cerr << "  lr: " << std::setprecision(6) << lr;
  std::cerr << "  loss: " << std::setprecision(6) << loss;
  std::cerr << "  eta: " << etah << "h" << etam << "m ";
  std::cerr << std::flush;
}

std::vector<int32_t> FastText::selectEmbeddings(int32_t cutoff) const {
  Vector norms(input_->m_);
  input_->l2NormRow(norms);
  std::vector<int32_t> idx(input_->m_, 0);
  std::iota(idx.begin(), idx.end(), 0);
  auto eosid = dict_->getId(Dictionary::EOS);
  std::sort(idx.begin(), idx.end(), [&norms, eosid](size_t i1, size_t i2) {
    return eosid == i1 || (eosid != i2 && norms[i1] > norms[i2]);
  });
  idx.erase(idx.begin() + cutoff, idx.end());
  return idx;
}

void FastText::quantize(std::shared_ptr<Args> qargs) {
  if (args_->model != model_name::sup) {
    throw std::invalid_argument(
        "For now we only support quantization of supervised models");
  }
  args_->input = qargs->input;
  args_->qout = qargs->qout;
  args_->output = qargs->output;

  if (qargs->cutoff > 0 && qargs->cutoff < input_->m_) {
    auto idx = selectEmbeddings(qargs->cutoff);
    dict_->prune(idx);
    std::shared_ptr<Matrix> ninput =
        std::make_shared<Matrix>(idx.size(), args_->dim);
    for (auto i = 0; i < idx.size(); i++) {
      for (auto j = 0; j < args_->dim; j++) {
        ninput->at(i, j) = input_->at(idx[i], j);
      }
    }
    input_ = ninput;
    if (qargs->retrain) {
      args_->epoch = qargs->epoch;
      args_->lr = qargs->lr;
      args_->thread = qargs->thread;
      args_->verbose = qargs->verbose;
      startThreads();
    }
  }

  qinput_ = std::make_shared<QMatrix>(*input_, qargs->dsub, qargs->qnorm);

  if (args_->qout) {
    qoutput_ = std::make_shared<QMatrix>(*output_, 2, qargs->qnorm);
  }

  quant_ = true;
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);
}

void FastText::supervised(Model& model, real lr,
                          const std::vector<int32_t>& line,
                          const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) return;
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  model.update(line, labels[i], lr);
}

void FastText::pctr(Model& model, real lr, const std::vector<int32_t>& line,
                    const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) return;
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  model.update(line, labels[i], lr);
}

void FastText::cbow(Model& model, real lr, const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}

// 只使用前面的词预测后面的词
void FastText::xcbow(Model& model, real lr, const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  // w at least 1
  for (int32_t w = 1; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c < 0; c++) {
      if (c != 0 && w + c >= 0) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}

void FastText::skipgram(Model& model, real lr,
                        const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

void FastText::test(std::istream& in, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;

  while (in.peek() != EOF) {
    dict_->getLine(in, line, labels, model_->rng);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->predict(line, k, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend();
           it++) {
        if (std::find(labels.begin(), labels.end(), it->second) !=
            labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  std::cout << "N"
            << "\t" << nexamples << std::endl;
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << "\t" << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << "\t" << precision / nlabels << std::endl;
  std::cerr << "Number of examples: " << nexamples << std::endl;
}

void FastText::pctrTest(std::istream& in, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;

  while (in.peek() != EOF) {
    dict_->getLine(in, line, labels, model_->rng);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->pctrPredict(line, k, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend();
           it++) {
        if (std::find(labels.begin(), labels.end(), it->second) !=
            labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  std::cout << "N"
            << "\t" << nexamples << std::endl;
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << "\t" << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << "\t" << precision / nlabels << std::endl;
  std::cerr << "Number of examples: " << nexamples << std::endl;
}

void FastText::predict(
    std::istream& in, int32_t k,
    std::vector<std::pair<real, std::string>>& predictions) const {
  std::vector<int32_t> words, labels;
  predictions.clear();
  dict_->getLine(in, words, labels, model_->rng);
  predictions.clear();
  if (words.empty()) return;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());
  std::vector<std::pair<real, int32_t>> modelPredictions;
  model_->predict(words, k, modelPredictions, hidden, output);
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend();
       it++) {
    predictions.push_back(
        std::make_pair(it->first, dict_->getLabel(it->second)));
  }
}

void FastText::pctrPredict(
    std::istream& in, int32_t k,
    std::vector<std::pair<real, std::string>>& predictions) const {
  std::vector<int32_t> words, labels;
  dict_->getLine(in, words, labels, model_->rng);
  predictions.clear();
  if (words.empty()) return;
  Vector hidden(args_->dim * 2);
  Vector output(dict_->nlabels());

  std::vector<std::pair<real, int32_t>> modelPredictions;
  model_->pctrPredict(words, k, modelPredictions, hidden, output);
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend();
       it++) {
    predictions.push_back(
        std::make_pair(it->first, dict_->getLabel(it->second)));
  }
}

void FastText::predict(std::istream& in, int32_t k, bool print_prob) {
  std::vector<std::pair<real, std::string>> predictions;
  while (in.peek() != EOF) {
    predictions.clear();
    predict(in, k, predictions);
    if (predictions.empty()) {
      std::cout << std::endl;
      continue;
    }
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << " ";
      }
      std::cout << it->second;
      if (print_prob) {
        std::cout << " " << exp(it->first);
      }
    }
    std::cout << std::endl;
  }
}

void FastText::pctrPredict(std::istream& in, int32_t k) {
  std::vector<std::pair<real, std::string>> predictions;
  while (in.peek() != EOF) {
    predictions.clear();
    pctrPredict(in, k, predictions);
    if (predictions.empty()) {
      std::cout << std::endl;
      continue;
    }
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << " ";
      }
      std::cout << it->second;
      std::cout << " " << exp(it->first);
    }
    std::cout << std::endl;
  }
}

void FastText::pctrPredict(
    const std::vector<std::string>& inputs, int32_t k,
    std::vector<std::pair<real, std::string>>& predictions) const {
  predictions.clear();
  std::vector<int32_t> words;
  for (auto& w : inputs) {
    int idx = dict_->getId(w);
    if (idx < 0) {
      continue;
    }
    words.push_back(idx);
  }

  if (words.empty()) return;
  Vector hidden(args_->dim * 2);
  Vector output(dict_->nlabels());

  std::vector<std::pair<real, int32_t>> modelPredictions;
  model_->pctrPredict(words, k, modelPredictions, hidden, output);
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend();
       it++) {
    predictions.push_back(
        std::make_pair(exp(it->first), dict_->getLabel(it->second)));
  }
}

void FastText::getSentenceVector(std::istream& in, fasttext::Vector& svec) {
  svec.zero();
  if (args_->model == model_name::sup) {
    std::vector<int32_t> line, labels;
    dict_->getLine(in, line, labels, model_->rng);
    for (int32_t i = 0; i < line.size(); i++) {
      addInputVector(svec, line[i]);
    }
    if (!line.empty()) {
      svec.mul(1.0 / line.size());
    }
  } else {
    Vector vec(args_->dim);
    std::string sentence;
    std::getline(in, sentence);
    std::istringstream iss(sentence);
    std::string word;
    int32_t count = 0;
    while (iss >> word) {
      getWordVector(vec, word);
      real norm = vec.norm();
      if (norm > 0) {
        vec.mul(1.0 / norm);
        svec.addVector(vec);
        count++;
      }
    }
    if (count > 0) {
      svec.mul(1.0 / count);
    }
  }
}

void FastText::ngramVectors(std::string word) {
  std::vector<int32_t> ngrams;
  std::vector<std::string> substrings;
  Vector vec(args_->dim);
  dict_->getSubwords(word, ngrams, substrings);
  for (int32_t i = 0; i < ngrams.size(); i++) {
    vec.zero();
    if (ngrams[i] >= 0) {
      if (quant_) {
        vec.addRow(*qinput_, ngrams[i]);
      } else {
        vec.addRow(*input_, ngrams[i]);
      }
    }
    std::cout << substrings[i] << " " << vec << std::endl;
  }
}

void FastText::precomputeWordVectors(Matrix& wordVectors) {
  Vector vec(args_->dim);
  wordVectors.zero();
  std::cerr << "Pre-computing word vectors...";
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    real norm = vec.norm();
    if (norm > 0) {
      wordVectors.addRow(vec, i, 1.0 / norm);
    }
  }
  std::cerr << " done." << std::endl;
}

void FastText::precomputeWordVectors(bool col_major) {
  if (!precomputed_) {
    if (!col_major && args_->maxn == 0) {
      wordVectors_ = input_;
    } else {
      wordVectors_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
      precomputeWordVectors(*wordVectors_);
      if (col_major) {
        wordVectors_->convertColMajor();
      }
    }
    precomputed_ = true;
  }
}

void FastText::precomputeSubsetVectors(bool col_major) {
  if (!precomputed_subset_) {
    // TODO 不考虑 subword
    subsetVectors_ = std::make_shared<Matrix>(subset_.size(), args_->dim);
    precomputeSubsetVectors(*subsetVectors_);
    if (col_major) {
      subsetVectors_->convertColMajor();
    }
    precomputed_subset_ = true;
  }
}

void FastText::precomputeSubsetVectors(Matrix& subsetVectors) {
  Vector vec(args_->dim);
  subsetVectors.zero();
  std::cerr << "Pre-computing word vectors...";
  for (int32_t i = 0; i < subset_.size(); ++i) {
    std::string word = dict_->getWord(subset_[i]);
    getWordVector(vec, word);
    real norm = vec.norm();
    if (norm > 0) {
      subsetVectors.addRow(vec, i, 1.0 / norm);
    }
  }
  std::cerr << " done." << std::endl;
}

void FastText::precomputeLabelsMatrix() {
  labels_matrix_ = std::make_shared<Matrix>(*output_);
  labels_matrix_->normlize();
  labels_matrix_->convertColMajor();  // for openblas
  precomputed_labels_matrix_ = true;
  std::cout << "Precompute labels matrix OK" << std::endl;
}

// wordVectors 首先要是 col major, 计算 wordVectors * queryVec,
// 结果保存在 output 中
void FastText::findNNHelper(const Matrix& wordVectors, const Vector& queryVec,
                            Vector& output) {
  assert(output.m_ == wordVectors.m_);
  int m = static_cast<int>(wordVectors.m_);
  int n = static_cast<int>(wordVectors.n_);
  real alpha = 1.0;
  real* a = wordVectors.data_;
  int lda = m;
  real* x = queryVec.data_;
  int incx = 1;
  real beta = 0.0;
  real* y = output.data_;
  int incy = 1;

  cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta,
              y, incy);
}

void FastText::findNN(const Matrix& wordVectors, const Vector& queryVec,
                      int32_t k, int32_t t,
                      const std::unordered_set<std::string>& banSet,
                      bool is_label) {
  std::vector<std::pair<real, std::string>> nn;
  findNN(wordVectors, queryVec, k, t, banSet, nn, is_label);
  for (auto& p : nn) {
    std::cout << p.second.substr(args_->label.size()) << " " << p.first
              << std::endl;
  }
}

void FastText::findNN(const Matrix& wordVectors, const Vector& queryVec,
                      int32_t k, int32_t t,
                      const std::unordered_set<std::string>& banSet,
                      std::vector<std::pair<real, std::string>>& nn,
                      bool is_label) {
  real queryNorm = queryVec.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }

  int sz = 0;
  if (is_label) {
    sz = dict_->nlabels();
  } else {
    sz = dict_->nwords();
  }
  std::vector<std::pair<real, int>> heap(sz);
  Vector output(sz);
  findNNHelper(wordVectors, queryVec, output);
  for (int32_t i = 0; i < sz; i++) {
    heap[i].first = output[i] / queryNorm;
    heap[i].second = i;
  }
  std::make_heap(heap.begin(), heap.end());

  int32_t i = 0;
  size_t poped = 0;
  while (i < k && heap.size() > 0) {
    auto& top = heap.front();
    std::string word;
    if (is_label) {
      word = dict_->getLabel(top.second);
    } else {
      word = dict_->getWord(top.second);
    }
    auto it = banSet.find(word);
    if (it == banSet.end()) {
      nn.push_back({top.first, word});
      i++;
    }
    pop_heap(heap.begin(), heap.end() - poped);
    ++poped;
  }
}

void FastText::findNNSubset(const Matrix& subsetVectors, const Vector& queryVec,
                            int32_t k, int32_t t,
                            const std::unordered_set<std::string>& banSet) {
  std::vector<std::pair<real, std::string>> nn;
  findNNSubset(subsetVectors, queryVec, k, t, banSet, nn);
  for (auto& p : nn) {
    std::cout << p.second << " " << p.first << std::endl;
  }
}

void FastText::findNNSubset(const Matrix& subsetVectors, const Vector& queryVec,
                            int32_t k, int32_t t,
                            const std::unordered_set<std::string>& banSet,
                            std::vector<std::pair<real, std::string>>& nn) {
  real queryNorm = queryVec.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }
  std::vector<std::pair<real, int>> heap(dict_->nwords());
  Vector output(subset_.size());
  findNNHelper(subsetVectors, queryVec, output);
  for (int32_t i = 0; i < subset_.size(); i++) {
    heap[i].first = output[i] / queryNorm;
    heap[i].second = subset_[i];
  }
  std::make_heap(heap.begin(), heap.end());

  int32_t i = 0;
  size_t poped = 0;
  while (i < k && heap.size() > 0) {
    auto& top = heap.front();

    auto& word = dict_->getWord(top.second);
    auto it = banSet.find(dict_->getWord(top.second));
    if (it == banSet.end()) {
      nn.push_back({top.first, word});
      i++;
    }
    pop_heap(heap.begin(), heap.end() - poped);
    ++poped;
  }
}

void FastText::nn(int32_t k, int32_t t) {
  std::string queryWord;
  Vector queryVec(args_->dim);
  precomputeWordVectors(true);

  std::unordered_set<std::string> banSet;
  while (std::cin >> queryWord) {
    banSet.clear();
    banSet.insert(queryWord);
    getWordVector(queryVec, queryWord);
    findNN(*wordVectors_, queryVec, k, t, banSet);
  }
}

void FastText::nnlabels(int32_t k, int32_t t) {
  precomputeLabelsMatrix();

  std::string queryLabel;
  Vector queryVec(args_->dim);
  std::unordered_set<std::string> banSet;
  while (std::cin >> queryLabel) {
    queryLabel = args_->label + queryLabel;
    banSet.clear();
    banSet.insert(queryLabel);
    getLabelVector(queryVec, queryLabel);
    findNN(*labels_matrix_, queryVec, k, t, banSet, true);
  }
}

void FastText::nnsubset(int32_t k, int32_t t) {
  std::string queryWord;
  Vector queryVec(args_->dim);
  // TODO subset vectors
  precomputeSubsetVectors(true);

  std::unordered_set<std::string> banSet;
  while (std::cin >> queryWord) {
    banSet.clear();
    banSet.insert(queryWord);
    getWordVector(queryVec, queryWord);
    findNNSubset(*subsetVectors_, queryVec, k, t, banSet);
  }
}

// api
void FastText::nnwithquery(const std::string& queryWord, int32_t k, int32_t t,
                           std::vector<std::pair<real, std::string>>& nn) {
  assert(precomputed_ == true);
  nn.clear();
  Vector queryVec(args_->dim);
  std::unordered_set<std::string> banSet;

  banSet.clear();
  banSet.insert(queryWord);
  getWordVector(queryVec, queryWord);
  findNN(*wordVectors_, queryVec, k, t, banSet, nn);
}

void FastText::nnlabelswithquery(
    const std::string& queryLabel, int32_t k, int32_t t,
    std::vector<std::pair<real, std::string>>& nnlabels) {
    auto query = args_->label + queryLabel;
    std::unordered_set<std::string> banSet;
    banSet.clear();
    banSet.insert(query);
    Vector queryVec(args_->dim);
    getLabelVector(queryVec, query);
    findNN(*labels_matrix_, queryVec, k, t, banSet, nnlabels, true);
    for (auto &p : nnlabels) {
      p.second = p.second.substr(args_->label.size());
    }
}

void FastText::nnsubsetwithquery(
    const std::string& query, int32_t k, int32_t t,
    std::vector<std::pair<real, std::string>>& nn) {
  assert(precomputed_subset_ == true);
  Vector queryVec(args_->dim);
  std::unordered_set<std::string> banSet;
  banSet.clear();
  banSet.insert(query);
  getWordVector(queryVec, query);

  findNNSubset(*subsetVectors_, queryVec, k, t, banSet, nn);
}

void FastText::nnwithvector(const std::vector<std::string>& inputs, int32_t k,
                            int32_t t,
                            std::vector<std::pair<real, std::string>>& nn) {
  assert(precomputed_ == true);
  if (inputs.size() == 0) {
    return;
  }

  std::unordered_set<std::string> banSet;
  banSet.clear();
  Vector queryVec(args_->dim);
  Vector meanVec(args_->dim);
  meanVec.zero();
  for (auto& input : inputs) {
    banSet.insert(input);
    getWordVector(queryVec, input);
    meanVec.addVector(queryVec);
  }

  meanVec.mul(1.0 / inputs.size());
  findNN(*wordVectors_, meanVec, k, t, banSet, nn);
}

void FastText::nnlsh(int32_t k, int32_t t) {
  std::string queryWord;
  Vector queryVec(args_->dim);
  precomputeWordVectors();
  std::unordered_set<std::string> banSet;
  while (std::cin >> queryWord) {
    banSet.clear();
    banSet.insert(queryWord);
    getWordVector(queryVec, queryWord);

    findNN(*wordVectors_, queryVec, k, t, banSet);
  }
}

void FastText::analogies(int32_t k) {
  std::string word;
  Vector buffer(args_->dim), query(args_->dim);
  precomputeWordVectors();
  std::unordered_set<std::string> banSet;
  std::cout << "Query triplet (A - B + C)? ";
  while (true) {
    banSet.clear();
    query.zero();
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, 1.0);
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, -1.0);
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, 1.0);

    findNN(*wordVectors_, query, k, 0, banSet);
    std::cout << "Query triplet (A - B + C)? ";
  }
}

void FastText::trainThread(int32_t threadId) {
  std::ifstream ifs(args_->input);
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model model(input_, output_, args_, threadId);
  if (args_->model == model_name::sup || args_->model == model_name::pctr) {
    model.setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model.setTargetCounts(dict_->getCounts(entry_type::word));
  }

  const int64_t ntokens = dict_->ntokens();
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels;
  while (tokenCount < args_->epoch * ntokens) {
    real progress = real(tokenCount) / (args_->epoch * ntokens);
    real lr = args_->lr * (1.0 - progress);
    if (args_->model == model_name::sup) {
      localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
      supervised(model, lr, line, labels);
    } else if (args_->model == model_name::pctr) {
      localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
      pctr(model, lr, line, labels);
    } else if (args_->model == model_name::cbow) {
      localTokenCount += dict_->getLine(ifs, line, model.rng);
      cbow(model, lr, line);
    } else if (args_->model == model_name::xcbow) {
      localTokenCount += dict_->getLine(ifs, line, model.rng);
      xcbow(model, lr, line);
    } else if (args_->model == model_name::sg) {
      localTokenCount += dict_->getLine(ifs, line, model.rng);
      skipgram(model, lr, line);
    }
    if (localTokenCount > args_->lrUpdateRate) {
      tokenCount += localTokenCount;
      localTokenCount = 0;
      if (threadId == 0 && args_->verbose > 1) {
        printInfo(progress, model.getLoss());
      }
    }
  }
  if (threadId == 0 && args_->verbose > 0) {
    printInfo(1.0, model.getLoss());
    std::cerr << std::endl;
  }
  ifs.close();
}

void FastText::loadVectors(std::string filename) {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<Matrix> mat;  // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  in >> n >> dim;
  if (dim != args_->dim) {
    std::cerr << "Dimension of pretrained vectors does not match -dim option"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  mat = std::make_shared<Matrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->add(word);
    for (size_t j = 0; j < dim; j++) {
      in >> mat->data_[i * dim + j];
    }
  }
  in.close();

  dict_->threshold(1, 0);
  input_ =
      std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
  input_->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++) {
    int32_t idx = dict_->getId(words[i]);
    if (idx < 0 || idx >= dict_->nwords()) continue;
    for (size_t j = 0; j < dim; j++) {
      input_->data_[idx * dim + j] = mat->data_[i * dim + j];
    }
  }
}

void FastText::train(std::shared_ptr<Args> args) {
  args_ = args;
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    std::cerr << "Cannot use stdin for training!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    std::cerr << "Input file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  dict_->readFromFile(ifs);
  ifs.close();

  if (args_->pretrainedVectors.size() != 0) {
    loadVectors(args_->pretrainedVectors);
  } else {
    input_ =
        std::make_shared<Matrix>(dict_->nwords() + args_->bucket, args_->dim);
    input_->uniform(1.0 / args_->dim);
  }

  if (args_->model == model_name::sup) {
    output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
  } else if (args->model == model_name::pctr) {
    output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim * 2);
  } else {
    output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
  }
  output_->zero();
  startThreads();
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
}

void FastText::startThreads() {
  start = clock();
  tokenCount = 0;
  if (args_->thread > 1) {
    std::vector<std::thread> threads;
    for (int32_t i = 0; i < args_->thread; i++) {
      threads.push_back(std::thread([=]() { trainThread(i); }));
    }
    for (auto it = threads.begin(); it != threads.end(); ++it) {
      it->join();
    }
  } else {
    trainThread(0);
  }
}

int FastText::getDimension() const { return args_->dim; }
bool FastText::isQuant() const { return quant_; }

void FastText::ComputeSimilarity(const std::string& target,
                                 const std::vector<std::string>& queries,
                                 std::vector<float>& similarity) {
  assert(precomputed_ == true);
  Vector targetVec(args_->dim);
  getWordVector(targetVec, target);
  real targetNorm = targetVec.norm();
  if (std::abs(targetNorm) < 1e-8) {
    targetNorm = 1;
  }
  Vector vec(args_->dim);
  for (size_t i = 0; i < queries.size(); i++) {
    vec.zero();
    getWordVector(vec, queries[i]);
    real vecNorm = vec.norm();
    if (std::abs(vecNorm) < 1e-8) {
      vecNorm = 1;
    }
    real dp = targetVec.dot(vec);
    similarity.push_back(dp / targetNorm / vecNorm);
  }
}

void FastText::ComputeSimilarityMean(
    const std::vector<std::string>& target,
    const std::vector<std::vector<std::string>>& queries,
    std::vector<float>& similarity) {
  assert(precomputed_ == true);
  similarity.clear();

  Vector targetVec(args_->dim);
  targetVec.zero();
  Vector tmpVec(args_->dim);
  for (const auto& w : target) {
    getWordVector(tmpVec, w);
    targetVec.addVector(tmpVec);
  }
  targetVec.mul(1.0 / target.size());
  real targetNorm = targetVec.norm();
  if (std::abs(targetNorm) < 1e-8) {
    targetNorm = 1;
  }

  Vector queryVec(args_->dim);
  for (const auto& query : queries) {
    if (query.empty()) {
      similarity.push_back(0.0);
    }

    queryVec.zero();
    for (const auto& w : query) {
      getWordVector(tmpVec, w);
      queryVec.addVector(tmpVec);
    }
    queryVec.mul(1.0 / query.size());
    real queryNorm = queryVec.norm();
    if (std::abs(queryNorm) < 1e-8) {
      queryNorm = 1;
    }
    // computer similarity
    real dp = targetVec.dot(queryVec);
    similarity.push_back(dp / targetNorm / queryNorm);
  }
}

void FastText::XcbowPredict(
    const std::vector<std::string>& inputs, int k,
    std::vector<std::pair<real, std::string>>& predicts) {
  std::unordered_set<std::string> banSet;
  Vector meanVec(args_->dim);
  meanVec.zero();
  for (const auto& input : inputs) {
    banSet.insert(input);
    // TODO maxn should be 0 in this optimized method
    int idx = dict_->getId(input);
    if (idx < 0) {
      continue;
    }
    meanVec.addRow(*wordVectors_, idx);
  }
  meanVec.mul(1.0 / inputs.size());
  FindXcbowPredicts(meanVec, k, banSet, predicts);
}

void FastText::XcbowPredictLSH(
    const std::vector<std::string>& inputs, int k,
    std::vector<std::pair<real, std::string>>& predicts) {
  std::unordered_set<std::string> banSet;
  banSet.clear();
  Vector queryVec(args_->dim);
  Vector meanVec(args_->dim);
  meanVec.zero();
  for (auto& input : inputs) {
    banSet.insert(input);
    getWordVector(queryVec, input);
    meanVec.addVector(queryVec);
  }
  meanVec.mul(1.0 / inputs.size());

  FindXcbowPredictsLSH(meanVec, k, banSet, predicts);
}

void FastText::FindXcbowPredicts(
    Vector& meanVec, int32_t k, const std::unordered_set<std::string>& banSet,
    std::vector<std::pair<real, std::string>>& predicts) {
  Vector output(dict_->nwords());
  model_->computeOutput(meanVec, output);
  std::vector<std::pair<real, int>> heap;
  for (int32_t i = 0; i < dict_->nwords(); ++i) {
    heap.push_back(std::make_pair(output[i], i));
  }
  std::make_heap(heap.begin(), heap.end());

  int32_t i = 0;
  size_t poped = 0;
  while (i < k && poped < heap.size()) {
    auto& top = heap.front();
    pop_heap(heap.begin(), heap.end() - poped);
    ++poped;

    auto& word = dict_->getWord(top.second);
    auto it = banSet.find(word);
    if (it == banSet.end()) {
      predicts.push_back(std::make_pair(top.first, word));
      i++;
    }
  }
}

void FastText::FindXcbowPredictsLSH(
    Vector& meanVec, int32_t k, const std::unordered_set<std::string>& banSet,
    std::vector<std::pair<real, std::string>>& predicts) {
  std::vector<int> closest;
  std::vector<float> dist;
  size_t n = lsh_idx2key.size();
  annoy_index->get_nns_by_vector(meanVec.data_, k + banSet.size(), n, &closest,
                                 &dist);

  for (int i = 0; i < closest.size(); ++i) {
    if (predicts.size() == k) {
      break;
    }
    auto& key = lsh_idx2key[closest[i]];
    if (banSet.find(key) != banSet.end()) {
      continue;
    }
    predicts.emplace_back(dist[i], key);
  }
}

void FastText::feedsubset(const std::vector<std::string>& subset) {
  for (auto& s : subset) {
    int32_t id = dict_->getId(s);
    if (id < 0) {
      std::cerr << "[E] feedsubset: " << s << "not in dict." << std::endl;
      exit(-1);
    }
    subset_.push_back(id);
  }

  if (args_->model == model_name::xcbow) {
    output_->shrinkSubset(subset_);
    output_->convertColMajor();
  }
}

static bool run_cmd(char* cmd) {
  pid_t pid;
  char sh[4] = "sh";
  char arg[4] = "-c";
  char* argv[] = {sh, arg, cmd, NULL};
  std::cerr << "Run command: " << cmd << std::endl;
  int status = posix_spawn(&pid, "/bin/sh", NULL, NULL, argv, environ);
  if (status == 0) {
    std::cerr << "Child pid: " << pid << std::endl;
    if (waitpid(pid, &status, 0) != -1) {
      std::cerr << "Child exited with status " << status << std::endl;
    } else {
      std::cerr << "Child exited with status " << status
                << ", errmsg = " << strerror(errno) << std::endl;
      return false;
    }
  } else {
    std::cerr << "posix_spawn failed, errmsg = " << strerror(status)
              << std::endl;
    return false;
  }
  return true;
}

static void OpenFileRead(const std::string& file, std::ifstream& ifs) {
  ifs.open(file);
  if (!ifs.is_open()) {
    std::cerr << "open file [" << file << "] to read failed." << std::endl;
    exit(-1);
  }
}

static void OpenFileWrite(const std::string& file, std::ofstream& ofs) {
  ofs.open(file);
  if (!ofs.is_open()) {
    std::cerr << "open file [" << file << "] to write failed." << std::endl;
    exit(-1);
  }
}

void FastText::multiNNThread(const std::string& queryfile, int k,
                             bool usesubset) {
  std::string resultfile = queryfile + ".result";
  std::ifstream ifs;
  std::ofstream ofs;
  OpenFileRead(queryfile, ifs);
  OpenFileWrite(resultfile, ofs);

  std::string query;
  std::vector<std::pair<real, std::string>> nn;
  while (!ifs.eof()) {
    std::getline(ifs, query);
    if (query.empty()) {
      continue;
    }

    nn.clear();
    if (usesubset) {
      nnsubsetwithquery(query, k, 0, nn);
    } else {
      nnwithquery(query, k, 0, nn);
    }
    for (auto& p : nn) {
      ofs.write(p.second.data(), p.second.size());
      ofs.write(" ", 1);
      auto s = std::to_string(p.first);
      ofs.write(s.data(), s.size());
      ofs.write("\n", 1);
    }
  }
  ifs.close();
  ofs.close();
}

void FastText::multinn(const std::string& queryfile, int nthreads, int k,
                       bool usesubset) {
  static char cmd[65536];
  std::string command = "split -a 3 -d -n l/" + std::to_string(nthreads) + " " +
                        queryfile + " " + queryfile + ".";
  memcpy(cmd, command.data(), command.size());
  cmd[command.size()] = '\0';
  if (!run_cmd(cmd)) {
    exit(-1);
  }

  if (usesubset) {
    precomputeSubsetVectors(true);
  } else {
    precomputeWordVectors(true);
  }

  std::vector<std::thread> threads;
  char suffix[4];
  std::string catfile;
  std::string rmfile;
  for (int i = 0; i < nthreads; ++i) {
    snprintf(suffix, 4, "%03d", i);
    auto name = queryfile + "." + suffix;
    catfile += name + ".result";
    catfile += " ";
    rmfile += name;
    rmfile += " ";
    std::cerr << "split query file name: " << name << std::endl;
    threads.emplace_back(&FastText::multiNNThread, this, name, k, usesubset);
  }

  for (int i = 0; i < nthreads; ++i) {
    threads[i].join();
  }

  command = "cat " + catfile + " " + " > " + queryfile + ".result";
  memcpy(cmd, command.data(), command.size());
  cmd[command.size()] = '\0';
  if (!run_cmd(cmd)) {
    exit(-1);
  }

  command = "rm " + catfile + " " + rmfile;
  memcpy(cmd, command.data(), command.size());
  cmd[command.size()] = '\0';
  if (!run_cmd(cmd)) {
    exit(-1);
  }

  std::cerr << "multi nn done." << std::endl;
}

void FastText::XcbowPredictSubset(
    const std::vector<std::string>& inputs, int k,
    std::vector<std::pair<real, std::string>>& predicts) {
  Vector meanVec(args_->dim);
  Vector output(subset_.size());
  std::vector<std::pair<real, int>> heap(subset_.size());
  XcbowPredictSubsetOpt(inputs, k, predicts, meanVec, output, heap);
}

void FastText::FindXcbowPredictsSubsetOpt(
    Vector& meanVec, int32_t k, const std::unordered_set<std::string>& banSet,
    std::vector<std::pair<real, std::string>>& predicts, Vector& output,
    std::vector<std::pair<real, int>>& heap) {
  model_->computeOutput(meanVec, output);
  for (size_t i = 0; i < subset_.size(); ++i) {
    heap[i].first = output[i];
    heap[i].second = subset_[i];
  }
  std::make_heap(heap.begin(), heap.end());

  int32_t i = 0;
  size_t poped = 0;
  while (i < k && poped < heap.size()) {
    auto& top = heap.front();

    auto& word = dict_->getWord(top.second);
    auto it = banSet.find(word);
    if (it == banSet.end()) {
      predicts.push_back(std::make_pair(top.first, word));
      i++;
    }

    pop_heap(heap.begin(), heap.end() - poped);
    ++poped;
  }
}

void FastText::XcbowPredictSubsetOpt(
    const std::vector<std::string>& inputs, int k,
    std::vector<std::pair<real, std::string>>& predicts, Vector& meanVec,
    Vector& output, std::vector<std::pair<real, int>>& heap) {
  assert(meanVec.m_ == args_->dim);
  assert(output.m_ == subset_.size());
  assert(heap.size() == subset_.size());
  meanVec.zero();
  std::unordered_set<std::string> banSet;
  for (auto& p : predicts) {
    banSet.insert(p.second);
  }

  for (auto& input : inputs) {
    banSet.insert(input);
    // TODO maxn should be 0 in this optimized method
    int idx = dict_->getId(input);
    if (idx < 0) {
      continue;
    }
    meanVec.addRow(*wordVectors_, idx);
  }
  meanVec.mul(1.0 / inputs.size());
  FindXcbowPredictsSubsetOpt(meanVec, k, banSet, predicts, output, heap);
}

void FastText::LoadLSH(const std::string& lsh_build_file,
                       const std::string& lsh_dict_file) {
  int dim = args_->dim;
  annoy_index =
      std::make_shared<AnnoyIndex<int, float, Euclidean, Kiss32Random>>(dim);
  if (!annoy_index->load(lsh_build_file.c_str())) {
    std::cerr << "load lsh failed." << std::endl;
    abort();
  }

  std::ifstream ifs(lsh_dict_file);
  if (!ifs.is_open()) {
    std::cerr << "open lsh dict file failed." << std::endl;
    abort();
  }
  std::string key;
  lsh_idx2key.clear();
  lsh_idx2key.clear();
  while (!ifs.eof()) {
    std::getline(ifs, key);
    if (key.empty()) {
      continue;
    }
    lsh_key2idx[key] = lsh_idx2key.size();
    lsh_idx2key.push_back(key);
  }
}
}

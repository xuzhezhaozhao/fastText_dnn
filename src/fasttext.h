/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_FASTTEXT_H
#define FASTTEXT_FASTTEXT_H

#define FASTTEXT_VERSION 12 /* Version 1b */
#define FASTTEXT_FILEFORMAT_MAGIC_INT32 793712314

#include <time.h>

#include <atomic>
#include <memory>
#include <set>
#include <unordered_set>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "model.h"
#include "qmatrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

#include "annoy/src/annoylib.h"
#include "annoy/src/kissrandom.h"

namespace fasttext {

class FastText {
 private:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;

  std::shared_ptr<Matrix> input_;
  std::shared_ptr<Matrix> output_;
  std::shared_ptr<Matrix> labels_matrix_;

  std::shared_ptr<QMatrix> qinput_;
  std::shared_ptr<QMatrix> qoutput_;

  std::shared_ptr<Model> model_;

  std::atomic<int64_t> tokenCount;
  clock_t start;
  void signModel(std::ostream &);
  bool checkModel(std::istream &);

  bool quant_;
  int32_t version;

  bool precomputed_ = false;
  bool precomputed_subset_ = false;
  bool precomputed_labels_matrix_ = false;

  // nn 模式为 col major, predict 模式为 row major
  std::shared_ptr<Matrix> wordVectors_;
  std::shared_ptr<Matrix> subsetVectors_;

  std::vector<int32_t> subset_;
  std::shared_ptr<AnnoyIndex<int, float, Euclidean, Kiss32Random>> annoy_index;
  std::vector<std::string> lsh_idx2key;
  std::unordered_map<std::string, int> lsh_key2idx;

  void startThreads();

 public:
  FastText();

  int32_t getWordId(const std::string &) const;
  int32_t getSubwordId(const std::string &) const;
  FASTTEXT_DEPRECATED(
      "getVector is being deprecated and replaced by getWordVector.")
  void getVector(Vector &, const std::string &) const;
  void getWordVector(Vector &, const std::string &) const;
  void getLabelVector(Vector &, const std::string &) const;
  void getSubwordVector(Vector &, const std::string &) const;
  void addInputVector(Vector &, int32_t) const;
  inline void getInputVector(Vector &vec, int32_t ind) {
    vec.zero();
    addInputVector(vec, ind);
  }
  void addLabelVector(Vector& vec, int32_t ind) const;

  const Args getArgs() const;
  std::shared_ptr<const Dictionary> getDictionary() const;
  std::shared_ptr<const Matrix> getInputMatrix() const;
  std::shared_ptr<const Matrix> getOutputMatrix() const;
  void saveVectors();
  void saveModel(const std::string);
  void saveOutput();
  void saveModel();
  void loadModel(std::istream &);
  void loadModel(const std::string &);
  void printInfo(real, real);

  void supervised(Model &, real, const std::vector<int32_t> &,
                  const std::vector<int32_t> &);
  void pctr(Model &, real, const std::vector<int32_t> &,
            const std::vector<int32_t> &);
  void cbow(Model &, real, const std::vector<int32_t> &);
  void xcbow(Model &, real, const std::vector<int32_t> &);
  void skipgram(Model &, real, const std::vector<int32_t> &);
  std::vector<int32_t> selectEmbeddings(int32_t) const;
  void getSentenceVector(std::istream &, Vector &);
  void quantize(std::shared_ptr<Args>);
  void test(std::istream &, int32_t);
  void pctrTest(std::istream &, int32_t);
  void predict(std::istream &, int32_t, bool);
  void predict(std::istream &, int32_t,
               std::vector<std::pair<real, std::string>> &) const;

  void pctrPredict(std::istream &, int32_t);
  void pctrPredict(std::istream &, int32_t,
                   std::vector<std::pair<real, std::string>> &) const;

  void pctrPredict(const std::vector<std::string> &, int32_t,
                   std::vector<std::pair<real, std::string>> &) const;

  void ngramVectors(std::string);
  void precomputeWordVectors(Matrix &);
  void precomputeWordVectors(bool col_major = false);

  void precomputeSubsetVectors(Matrix &);
  void precomputeSubsetVectors(bool col_major = false);

  void precomputeLabelsMatrix();

  // 计算点乘
  void findNNHelper(const Matrix &wordVectors, const Vector &queryVec,
                    Vector &ouput);

  void findNN(const Matrix &, const Vector &, int32_t, int32_t,
              const std::unordered_set<std::string> &, bool is_label = false);
  void findNN(const Matrix &, const Vector &, int32_t, int32_t,
              const std::unordered_set<std::string> &,
              std::vector<std::pair<real, std::string>> &, bool is_label = false);
  void findNNSubset(const Matrix &, const Vector &, int32_t, int32_t,
                    const std::unordered_set<std::string> &);
  void findNNSubset(const Matrix &, const Vector &, int32_t, int32_t,
                    const std::unordered_set<std::string> &,
                    std::vector<std::pair<real, std::string>> &);
  void nn(int32_t, int32_t);
  // nn for labels
  void nnlabels(int32_t, int32_t);
  void nnsubset(int32_t, int32_t);
  void nnwithquery(const std::string &, int32_t, int32_t,
                   std::vector<std::pair<real, std::string>> &);
  void nnlabelswithquery(const std::string &, int32_t, int32_t,
                   std::vector<std::pair<real, std::string>> &);
  void nnlsh(int32_t, int32_t);
  void nnwithvector(const std::vector<std::string> &, int32_t, int32_t,
                    std::vector<std::pair<real, std::string>> &);
  void multinn(const std::string &queryfile, int threads, int k,
               bool usesubset);
  void multiNNThread(const std::string &queryfile, int k, bool usesubset);

  void feedsubset(const std::vector<std::string> &subset);
  void nnsubsetwithquery(const std::string &query, int32_t k, int32_t t,
                         std::vector<std::pair<real, std::string>> &nn);

  void analogies(int32_t);
  void trainThread(int32_t);
  void train(std::shared_ptr<Args>);

  void loadVectors(std::string);
  int getDimension() const;
  bool isQuant() const;

  void ComputeSimilarity(const std::string &target,
                         const std::vector<std::string> &queries,
                         std::vector<float> &similarity);

  void ComputeSimilarityMean(
      const std::vector<std::string> &target,
      const std::vector<std::vector<std::string>> &queries,
      std::vector<float> &similarity);

  void XcbowPredict(const std::vector<std::string> &inputs, int k,
                    std::vector<std::pair<real, std::string>> &predicts);
  void XcbowPredictLSH(const std::vector<std::string> &inputs, int k,
                       std::vector<std::pair<real, std::string>> &predicts);
  void FindXcbowPredicts(Vector &meanVec, int32_t k,
                         const std::unordered_set<std::string> &banSet,
                         std::vector<std::pair<real, std::string>> &predicts);
  void FindXcbowPredictsLSH(
      Vector &meanVec, int32_t k, const std::unordered_set<std::string> &banSet,
      std::vector<std::pair<real, std::string>> &predicts);
  void XcbowPredictSubset(const std::vector<std::string> &inputs, int k,
                          std::vector<std::pair<real, std::string>> &predicts);

  void XcbowPredictSubsetOpt(
      const std::vector<std::string> &inputs, int k,
      std::vector<std::pair<real, std::string>> &predicts, Vector &meanVec,
      Vector &output, std::vector<std::pair<real, int>> &heap);

  void FindXcbowPredictsSubset(
      Vector &meanVec, int32_t k, const std::unordered_set<std::string> &banSet,
      std::vector<std::pair<real, std::string>> &predicts);
  void FindXcbowPredictsSubsetOpt(
      Vector &meanVec, int32_t k, const std::unordered_set<std::string> &banSet,
      std::vector<std::pair<real, std::string>> &predicts, Vector &output,
      std::vector<std::pair<real, int>> &heap);

  void LoadLSH(const std::string &lsh_build_file,
               const std::string &lsh_dict_file);

  int dim() const { return args_->dim; }
  size_t subset_size() const { return subset_.size(); }
};

}  // namespace fasttext
#endif

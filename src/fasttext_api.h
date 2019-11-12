#ifndef FASTTEXT_FASTTEXT_API_H
#define FASTTEXT_FASTTEXT_API_H

#include <string>
#include <vector>

#include "fasttext.h"
#include "real.h"

namespace fasttext {

class FastTextApi {
 public:
  FastTextApi();

  struct SkipgramArgs {};

  struct CbowArgs {};
  struct SupervisedArgs {
    std::string input;
    std::string output;
    int32_t minCount = 1;
    int32_t minCountLabel = 0;
    int32_t wordNgrams = 1;
    int32_t bucket = 2000000;
    int32_t minn = 0;
    int32_t maxn = 0;
    double t = 0.0001;
    std::string label = "__label__";

    double lr = 0.1;
    int32_t lrUpdateRate = 100;
    int32_t dim = 100;
    int32_t ws = 5;
    int32_t epoch = 5;
    int32_t neg = 5;
    std::string loss = "softmax";
    int32_t thread = 1;

    // don't support
    // bool pretrainedVectors = false;
    // bool saveOutput = false;

    int32_t cutoff = 0;
    int32_t retrain = 0;
    int32_t qnorm = 0;
    int32_t qout = 0;
    int32_t dsub = 2;
  };

  struct TestArgs {
    std::string model;
    std::string test_data;
    int32_t k = 1;
  };

  struct QuantizeArgs {};
  struct NNArgs {};
  struct AnalogiesArgs {};

  struct PredictArgs {
    std::string model;
    std::string test_data;
    int32_t k = 1;
  };

  struct PredictStringArgs {
    std::string model;
    std::string test_string;
    int32_t k = 1;
  };

  struct PredictProbArgs {};

  void Skipgram();
  void Cbow();
  void Supervised(const SupervisedArgs &args);
  void Test(const TestArgs &args);
  void Quantize();

  std::vector<std::pair<real, std::string>> NN(const std::string &query, int k);
  std::vector<std::pair<real, std::string>> NNLabels(const std::string &query,
                                                     int k);

  void Analogies();
  std::vector<std::pair<real, std::string>> Predict(const std::string &test,
                                                    int k = 1,
                                                    bool usejieba = true);
  void PredictProb();
  void predictString(const std::string &test, int k,
                     std::vector<std::pair<real, std::string>> &predictions,
                     bool usejieba = true);

  std::vector<real> GetSentenceVector(const std::string &sentence);

  void LoadModel(const std::string &model);

  // 调用 nn 相关的 API 需要将 col_major 设为 true
  void PrecomputeWordVectors(bool col_major = false);
  void PrecomputeLabelsMatrix();

  std::vector<float> ComputeSimilarity(const std::string &target,
                                       const std::vector<std::string> &queries);

  std::vector<float> ComputeSimilarityMean(
      const std::vector<std::string> &target,
      const std::vector<std::vector<std::string>> &queries);

  // 对输入求加权平均
  std::vector<std::pair<real, std::string>> NNWithVector(
      const std::vector<std::string> &inputs, int k);

  std::vector<std::pair<real, std::string>> XcbowPredict(
      const std::vector<std::string> &inputs, int k);
  std::vector<std::pair<real, std::string>> XcbowPredictSubset(
      const std::vector<std::string> &inputs, int k);

  void XcbowPredictSubset(const std::vector<std::string> &inputs, int k,
                          std::vector<std::pair<real, std::string>> &predicts);

  std::vector<std::pair<real, std::string>> XcbowPredictSubsetOpt(
      const std::vector<std::string> &inputs, int k, Vector &meanVec,
      Vector &output, std::vector<std::pair<real, int>> &heap);

  std::vector<std::pair<real, std::string>> XcbowPredictLSH(
      const std::vector<std::string> &inputs, int k);

  std::vector<std::pair<real, std::string>> PCTRPredict(
      const std::vector<std::string> &inputs, int k);

  void FeedSubset(const std::vector<std::string> &subset);
  void FeedSubset(const std::string &subset_file);

  std::vector<std::pair<real, std::string>> NNSubset(const std::string &query,
                                                     int k);
  void LoadLSH(const std::string &lsh_build_file,
               const std::string &lsh_dict_file);

  int dim() const { return fasttext_.dim(); }
  size_t subset_size() const { return fasttext_.subset_size(); }

  const FastText& fasttext() { return fasttext_; }

 private:
  bool model_loaded_ = false;
  FastText fasttext_;
};
}  // namespace fasttext

#endif

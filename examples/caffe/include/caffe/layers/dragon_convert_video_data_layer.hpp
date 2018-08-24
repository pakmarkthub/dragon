/*
 *
 *  Copyright (c) 2015, Facebook, Inc. All rights reserved.
 *
 *  Licensed under the Creative Commons Attribution-NonCommercial 3.0
 *  License (the "License"). You may obtain a copy of the License at
 *  https://creativecommons.org/licenses/by-nc/3.0/.
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *
 */
 
#ifndef CAFFE_DRAGON_CONVERT_VIDEO_DATA_LAYER_HPP_
#define CAFFE_DRAGON_CONVERT_VIDEO_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DragonConvertVideoDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit DragonConvertVideoDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VideoData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {};
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleClips();

  Blob<Dtype> transformed_data_;

  vector<string> file_list_;
  vector<int> start_frm_list_;
  vector<int> label_list_;
  vector<vector<int> > multiple_label_list_;
  vector<int> shuffle_index_;
  int lines_id_;

  FILE *out_data_file_;
  FILE *out_label_file_;

  int limit_num_items_;
};


}  // namespace caffe

#endif  // CAFFE_DRAGON_CONVERT_VIDEO_DATA_LAYER_HPP_

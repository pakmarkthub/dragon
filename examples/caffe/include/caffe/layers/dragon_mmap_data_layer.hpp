#ifndef CAFFE_DRAGON_MMAP_DATA_LAYER_HPP_
#define CAFFE_DRAGON_MMAP_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DragonMmapDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit DragonMmapDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "DragonData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  virtual inline bool is_3d_data() {
      return this->data_depth_ > 0;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline size_t data_size_() {
      return this->size_per_data_item_() * this->data_length_;
  }

  virtual inline size_t size_per_data_item_() {
      return this->num_elements_per_data_item_() * sizeof(Dtype);
  }

  virtual inline size_t num_elements_per_data_item_() {
      size_t size = this->data_width_ * this->data_height_ * this->data_channels_;
      if (this->data_depth_ > 0)
          size *= this->data_depth_;
      return size;
  }

  Dtype* data_;
  Dtype* labels_;

  ::std::string data_source_;
  uint32_t data_width_;
  uint32_t data_height_;
  uint32_t data_depth_;
  uint32_t data_channels_;
  uint32_t data_length_;

  ::std::string labels_source_;
  uint32_t labels_length_;

  uint32_t batch_size_;

  uint32_t item_index_;
};

}  // namespace caffe

#endif  // CAFFE_DRAGON_MMAP_DATA_LAYER_HPP_

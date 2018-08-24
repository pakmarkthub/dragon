#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/dragon_convert_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DragonConvertDataLayer<Dtype>::DragonConvertDataLayer(const LayerParameter& param)
  : BaseDataLayer<Dtype>(param),
    transformed_data_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  std::string data_out = this->layer_param_.dragon_convert_data_param().data_out();
  std::string labels_out = this->layer_param_.dragon_convert_data_param().labels_out();

  out_data_file_ = fopen(data_out.c_str(), "w+");
  out_label_file_ = fopen(labels_out.c_str(), "w+");

  CHECK(out_data_file_) << "Cannot open/create " << data_out;
  CHECK(out_label_file_) << "Cannot open/create " << labels_out;
}

template <typename Dtype>
void DragonConvertDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  size_t write_ret = 0;
  long data_index = 0;
  while (cursor_->valid()) {
      datum.ParseFromString(cursor_->value());

      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      //fprintf(stderr, "===> top_shape: %d %d %d %d\n", top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
      this->transformed_data_.Reshape(top_shape);
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
      write_ret = fwrite(
         this->transformed_data_.cpu_data(),
         this->transformed_data_.count() * sizeof(Dtype),
         1,
         this->out_data_file_
      );
      CHECK_EQ(write_ret, 1) << "Cannot write to data.mem";
      // label
      if (this->output_labels_) {
          Dtype top_label = datum.label();
          write_ret = fwrite(
              &top_label,
              sizeof(Dtype),
              1,
              this->out_label_file_
          );
          CHECK_EQ(write_ret, 1) << "Cannot write to labels.mem";
      }
      cursor_->Next();
      if (data_index % 100 == 99)
          LOG(INFO) << "===> Converted data number: " << data_index + 1;
      ++data_index;
  }
  LOG(INFO) << "===> Converted data number: " << data_index;
  LOG(INFO) << "===> Finish converting the data.";
  exit(EXIT_SUCCESS);
}

INSTANTIATE_CLASS(DragonConvertDataLayer);
REGISTER_LAYER_CLASS(DragonConvertData);

}  // namespace caffe

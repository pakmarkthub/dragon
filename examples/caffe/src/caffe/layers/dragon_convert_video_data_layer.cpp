/*
 *
 *  Copyright (c) 2016, Facebook, Inc. All rights reserved.
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

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/dragon_convert_video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
DragonConvertVideoDataLayer<Dtype>::DragonConvertVideoDataLayer(const LayerParameter& param)
  : BaseDataLayer<Dtype>(param),
    transformed_data_() {

  std::string data_out = this->layer_param_.dragon_convert_data_param().data_out();
  std::string labels_out = this->layer_param_.dragon_convert_data_param().labels_out();

  this->limit_num_items_ = this->layer_param_.dragon_convert_data_param().limit_num_items();
  LOG(INFO) << "===> limit_num_items: " << this->limit_num_items_;

  out_data_file_ = fopen(data_out.c_str(), "w+");
  out_label_file_ = fopen(labels_out.c_str(), "w+");

  CHECK(out_data_file_) << "Cannot open/create " << data_out;
  CHECK(out_label_file_) << "Cannot open/create " << labels_out;
}

template <typename Dtype>
void DragonConvertVideoDataLayer<Dtype>::DataLayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int new_length = this->layer_param_.video_data_param().new_length();
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  string root_folder = this->layer_param_.video_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the list file
  const string& source = this->layer_param_.video_data_param().source();
  const bool use_temporal_jitter = this->layer_param_.video_data_param().use_temporal_jitter();
  const bool use_image = this->layer_param_.video_data_param().use_image();
  int sampling_rate = this->layer_param_.video_data_param().sampling_rate();
  const bool use_multiple_label = this->layer_param_.video_data_param().use_multiple_label();
  if (use_multiple_label) {
    CHECK(this->layer_param_.video_data_param().has_num_of_labels()) <<
    "number of labels must be set together with use multiple labels";

  }

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  int count = 0;
  string filename, labels;
  int start_frm, label;

  if (!use_multiple_label) {
    if ((!use_image) && use_temporal_jitter){
      while (infile >> filename >> label) {
        file_list_.push_back(filename);
        label_list_.push_back(label);
        shuffle_index_.push_back(count);
        count++;
      }
    } else {
      while (infile >> filename >> start_frm >> label) {
        file_list_.push_back(filename);
        start_frm_list_.push_back(start_frm);
        label_list_.push_back(label);
        shuffle_index_.push_back(count);
        count++;
  	  }
    }
  } else {
    if ((!use_image) && use_temporal_jitter){
      while (infile >> filename >> labels) {
        file_list_.push_back(filename);
        shuffle_index_.push_back(count);
        vector<int> label_set;
        int tmp_int;
        stringstream sstream(labels);
        while (sstream >> tmp_int) {
          label_set.push_back(tmp_int);
          if (sstream.peek() == ',')
            sstream.ignore();
        }
        multiple_label_list_.push_back(label_set);
        label_list_.push_back(label_set[0]);
        count++;
      }
    } else {
      while (infile >> filename >> start_frm >> labels) {
        file_list_.push_back(filename);
        start_frm_list_.push_back(start_frm);
        shuffle_index_.push_back(count);
        vector<int> label_set;
        int tmp_int;
        stringstream sstream(labels);
        while (sstream >> tmp_int) {
          label_set.push_back(tmp_int);
          if (sstream.peek() == ',')
            sstream.ignore();
        }
        multiple_label_list_.push_back(label_set);
        label_list_.push_back(label_set[0]);
        count++;
      }
    }
  }
  infile.close();

  if (this->layer_param_.video_data_param().shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      ShuffleClips();
  }

  if (count==0){
	  LOG(INFO) << "Failed to read the clip list" << std::endl;
  }
  lines_id_ = 0;
  LOG(INFO) << "A total of " << shuffle_index_.size() << " video chunks.";

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.video_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(shuffle_index_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }


  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  //string root_folder = video_data_param.root_folder();
  //const bool use_image = video_data_param.use_image();
  //const bool use_temporal_jitter = video_data_param.use_temporal_jitter();
  //int sampling_rate = video_data_param.sampling_rate();
  const int max_sampling_rate = video_data_param.max_sampling_rate();
  const bool use_sampling_rate_jitter = video_data_param.use_sampling_rate_jitter();
  //const bool show_data = video_data_param.show_data();

  //const bool use_multiple_label = this->layer_param_.video_data_param().use_multiple_label();
  if (use_multiple_label) {
    CHECK(this->layer_param_.video_data_param().has_num_of_labels()) <<
    "number of labels must be set together with use multiple labels";

  }

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  // Read a data point, and use it to initialize the top blob.
  VolumeDatum datum;
  bool read_status = false;
  const int dataset_size = shuffle_index_.size();

  while (!read_status) {
      int id = shuffle_index_[lines_id_];
      if (!use_image){
        if (use_temporal_jitter){
          read_status = ReadVideoToVolumeDatum((root_folder + file_list_[0]).c_str(), 0, label_list_[0],
                new_length, new_height, new_width, sampling_rate, &datum);
        } else {
          read_status = ReadVideoToVolumeDatum((root_folder + file_list_[id]).c_str(), start_frm_list_[id], label_list_[id],
                new_length, new_height, new_width, sampling_rate, &datum);
        }
      } else {
       // LOG(INFO) << "read video from " << file_list_[id].c_str();
       CHECK(ReadImageSequenceToVolumeDatum((root_folder + file_list_[id]).c_str(), start_frm_list_[id], label_list_[id],
                                  new_length, new_height, new_width, sampling_rate, &datum));
      }

      if (!read_status) {
          LOG(INFO) << "Skip " << (root_folder + file_list_[id]) << " " << start_frm_list_[id];
          this->lines_id_++;
          if (this->lines_id_ >= dataset_size) {
            // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            this->lines_id_ = 0;
            if (this->layer_param_.video_data_param().shuffle()){
                ShuffleClips();
            }
          }
      }
  }

  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);

  int num_items = dataset_size;
  if (this->limit_num_items_ > 0 && this->limit_num_items_ < dataset_size)
    num_items = this->limit_num_items_;

  // datum scales
  for (int item_id = 0; item_id < num_items; ++item_id) {
    // get a blob
    if (use_sampling_rate_jitter) {
      sampling_rate = caffe::caffe_rng_rand() % (max_sampling_rate) + 1;
    }
    CHECK_GT(dataset_size, lines_id_);
    int id = this->shuffle_index_[this->lines_id_];
    if (!use_image){
    	if (!use_temporal_jitter){
            read_status = ReadVideoToVolumeDatum((root_folder + this->file_list_[id]).c_str(), this->start_frm_list_[id],
            		this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
        }else{
        	read_status = ReadVideoToVolumeDatum((root_folder + this->file_list_[id]).c_str(), -1,
        			this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
        }
    } else {
        if (!use_temporal_jitter) {
        	read_status = ReadImageSequenceToVolumeDatum((root_folder + this->file_list_[id]).c_str(), this->start_frm_list_[id],
        			this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
        } else {
        	int num_of_frames = this->start_frm_list_[id];
        	int use_start_frame;
        	if (num_of_frames < new_length * sampling_rate){
        	    LOG(INFO) << "not enough frames; having " << num_of_frames;
        	    read_status = false;
        	} else {
        	    if (this->phase_ == TRAIN)
        	    	use_start_frame = caffe_rng_rand()%(num_of_frames-new_length*sampling_rate+1) + 1;
        	    else
        	    	use_start_frame = 0;
        	    read_status = ReadImageSequenceToVolumeDatum((root_folder + this->file_list_[id]).c_str(), use_start_frame,
        	    			    this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
        	}
        }
    }

    if (this->phase_ == TEST){
        CHECK(read_status) << "Testing must not miss any example";
    }

    if (!read_status) {
        this->lines_id_++;
        if (this->lines_id_ >= dataset_size) {
        	// We have reached the end. Restart from the first.
        	DLOG(INFO) << "Restarting data prefetching from start.";
        	this->lines_id_ = 0;
        	if (this->layer_param_.video_data_param().shuffle()){
        		ShuffleClips();
        	}
        }
        item_id--;
        continue;
    }

    // Apply transformations (mirror, crop...) to the video
   	this->data_transformer_->VideoTransform(datum, &(this->transformed_data_));
    size_t write_ret = fwrite(
       this->transformed_data_.cpu_data(),
       this->transformed_data_.count() * sizeof(Dtype),
       1,
       this->out_data_file_
    );
    CHECK_EQ(write_ret, 1) << "Cannot write to data.mem";


    if (this->output_labels_) {
        if (!use_multiple_label) {
          Dtype top_label = datum.label();
          write_ret = fwrite(
              &top_label,
              sizeof(Dtype),
              1,
              this->out_label_file_
          );
          CHECK_EQ(write_ret, 1) << "Cannot write to labels.mem";
        } else {
            LOG(INFO) << "Multiple labels are not supported.";
            exit(EXIT_FAILURE);
        }
    }

    // go to the next iter
    this->lines_id_++;
    if (lines_id_ >= dataset_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleClips();
      }
    }

    if (this->lines_id_ % 50 == 49)
        LOG(INFO) << "===> Converted data number: " << this->lines_id_ + 1 << " (from " << num_items << ")";
  }
  LOG(INFO) << "===> Converted data number: " << this->lines_id_;
  LOG(INFO) << "===> Finish converting the data.";
  exit(EXIT_SUCCESS);
}

template <typename Dtype>
void DragonConvertVideoDataLayer<Dtype>::ShuffleClips() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(shuffle_index_.begin(), shuffle_index_.end(), prefetch_rng);
}

INSTANTIATE_CLASS(DragonConvertVideoDataLayer);
REGISTER_LAYER_CLASS(DragonConvertVideoData);

}  // namespace caffe
#endif  // USE_OPENCV

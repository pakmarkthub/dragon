#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include <dragon.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/dragon_fread_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe 
{
    template <typename Dtype>
    DragonFreadDataLayer<Dtype>::DragonFreadDataLayer(const LayerParameter& param) 
        : BaseDataLayer<Dtype>(param)
    {
        this->data_source_ = this->layer_param_.dragon_data_param().data_source();
        this->data_width_ = this->layer_param_.dragon_data_param().width();
        this->data_height_ = this->layer_param_.dragon_data_param().height();
        this->data_depth_ = this->layer_param_.dragon_data_param().depth();
        this->data_channels_ = this->layer_param_.dragon_data_param().channels();
        this->data_length_ = this->layer_param_.dragon_data_param().length();

        this->labels_source_ = this->layer_param_.dragon_data_param().labels_source();
        this->labels_length_ = this->data_length_;

        this->batch_size_ = this->layer_param_.dragon_data_param().batch_size();

        if (typeid(Dtype) != typeid(float))
        {
            std::cerr << "We expect float but got " << typeid(Dtype).name() << " instead" << std::endl;
            abort();
        }

        this->data_fp_ = fopen(this->data_source_.c_str(), "rb");
        this->labels_fp_ = fopen(this->labels_source_.c_str(), "rb");

        CHECK(this->data_fp_ != NULL) << "Cannot open " << this->data_source_;
        CHECK(this->labels_fp_ != NULL) << "Cannot open " << this->labels_source_;
    }

    template <typename Dtype>
    void DragonFreadDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) 
    {
        if (this->is_3d_data())
        {
            vector<int> data_shape(5);
            data_shape[0] = this->batch_size_;
            data_shape[1] = this->data_channels_;
            data_shape[2] = this->data_depth_;
            data_shape[3] = this->data_height_;
            data_shape[4] = this->data_width_;

            top[0]->Reshape(data_shape);
        }
        else
            top[0]->Reshape(
                this->batch_size_,
                this->data_channels_,
                this->data_height_,
                this->data_width_
            );

        //CUDA_CHECK(cudaMallocManaged((void **)&this->data_, this->batch_size_ * this->size_per_data_item_() * 210));
        //CUDA_CHECK(cudaMallocManaged((void **)&this->data_, this->batch_size_ * this->size_per_data_item_()));
        this->data_ = (Dtype *)malloc(this->batch_size_ * this->size_per_data_item_());
        //this->data_ = (Dtype *)malloc(this->batch_size_ * this->size_per_data_item_() * 210);
        CHECK(this->data_ != NULL) << "Cannot malloc data_";
        //CHECK_EQ(fread(this->data_, this->batch_size_ * this->size_per_data_item_() * 200, 1, this->data_fp_), 1) << "Cannot read from " << this->data_source_;

        if (this->output_labels_)
        {
            vector<int> label_shape(1, this->batch_size_);
            top[1]->Reshape(label_shape);
            
            //CUDA_CHECK(cudaMallocManaged((void **)&this->labels_, this->batch_size_ * sizeof(Dtype) * 210));
            //CUDA_CHECK(cudaMallocManaged((void **)&this->labels_, this->batch_size_ * sizeof(Dtype)));
            this->labels_ = (Dtype *)malloc(this->batch_size_ * sizeof(Dtype));
            //this->labels_ = (Dtype *)malloc(this->batch_size_ * sizeof(Dtype) * 210);
            CHECK(this->labels_ != NULL) << "Cannot malloc labels_";
            //CHECK_EQ(fread(this->labels_, this->batch_size_ * sizeof(Dtype) * 200, 1, this->labels_fp_), 1) << "Cannot read from " << this->labels_source_;
        }

        this->item_index_ = 0;
    }

    template <typename Dtype>
    void DragonFreadDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) 
    {
        uint32_t batchsize = MIN(this->batch_size_, this->data_length_ - this->item_index_);
        if (this->is_3d_data())
        {
            vector<int> data_shape(5);
            data_shape[0] = batchsize;
            data_shape[1] = this->data_channels_;
            data_shape[2] = this->data_depth_;
            data_shape[3] = this->data_height_;
            data_shape[4] = this->data_width_;

            top[0]->Reshape(data_shape);
        }
        else
            top[0]->Reshape(
                batchsize,
                this->data_channels_,
                this->data_height_,
                this->data_width_
            );

        /*size_t block_size = (size_t)1 << 21;
        size_t total_read_size = batchsize * this->size_per_data_item_();
        while (total_read_size > 0)
        {
            CHECK_EQ(fread(this->data_, MIN(block_size, total_read_size), 1, this->data_fp_), 1) << "Cannot read from " << this->data_source_;
            total_read_size -= MIN(block_size, total_read_size);
        }*/
        CHECK_EQ(fread(this->data_, batchsize * this->size_per_data_item_(), 1, this->data_fp_), 1) << "Cannot read from " << this->data_source_;
        top[0]->set_cpu_data(this->data_);
        //top[0]->set_dragon_data(this->data_);
        //top[0]->set_cpu_data(this->data_ + this->item_index_ * this->num_elements_per_data_item_());
        //top[0]->set_dragon_data(this->data_ + this->item_index_ * this->num_elements_per_data_item_());

        if (this->output_labels_)
        {
            vector<int> label_shape(1, batchsize);
            top[1]->Reshape(label_shape);

            /*total_read_size = batchsize * sizeof(Dtype);
            while (total_read_size > 0)
            {
                CHECK_EQ(fread(this->labels_, MIN(block_size, total_read_size), 1, this->labels_fp_), 1) << "Cannot read from " << this->labels_source_;
                total_read_size -= MIN(block_size, total_read_size);
            }*/
            CHECK_EQ(fread(this->labels_, batchsize * sizeof(Dtype), 1, this->labels_fp_), 1) << "Cannot read from " << this->labels_source_;
            top[1]->set_cpu_data(this->labels_);
            //top[1]->set_dragon_data(this->labels_);
            //top[1]->set_cpu_data(this->labels_ + this->item_index_);
            //top[1]->set_dragon_data(this->labels_ + this->item_index_);
        }

        this->item_index_ += batchsize;
        if (this->item_index_ >= this->data_length_)
        {
            this->item_index_ = 0;

            rewind(this->data_fp_);
            rewind(this->labels_fp_);
        }
    }

    INSTANTIATE_CLASS(DragonFreadDataLayer);
    REGISTER_LAYER_CLASS(DragonFreadData);

}  // namespace caffe

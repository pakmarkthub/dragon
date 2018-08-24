#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/dragon_mmap_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe 
{
    template <typename Dtype>
    DragonMmapDataLayer<Dtype>::DragonMmapDataLayer(const LayerParameter& param) 
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
    }

    template <typename Dtype>
    void DragonMmapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) 
    {
        size_t datasize = this->data_size_();
        int data_fd;
        if ((data_fd = open(this->data_source_.c_str(), O_RDONLY | O_LARGEFILE)) < 0)
        {
            std::cerr << "Cannot open " << this->data_source_ << std::endl;
            abort();
        }
        this->data_ = (Dtype *)mmap(
            NULL,
            datasize, 
            PROT_READ,
            MAP_SHARED | MAP_NORESERVE,
            data_fd,
            0
        );
        if (this->data_ == MAP_FAILED)
        {
            std::cerr << "Cannot mmap " << this->data_source_ << std::endl;
            abort();
        }

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

        if (this->output_labels_)
        {
            size_t labelsize = sizeof(Dtype) * this->labels_length_;
            int label_fd;
            if ((label_fd = open(this->labels_source_.c_str(), O_RDONLY | O_LARGEFILE)) < 0)
            {
                std::cerr << "Cannot open " << this->labels_source_ << std::endl;
                abort();
            }
            this->labels_ = (Dtype *)mmap(
                NULL,
                labelsize, 
                PROT_READ,
                MAP_SHARED | MAP_NORESERVE,
                label_fd,
                0
            );
            if (this->labels_ == MAP_FAILED)
            {
                std::cerr << "Cannot mmap " << this->labels_source_ << std::endl;
                abort();
            }
            vector<int> label_shape(1, this->batch_size_);
            top[1]->Reshape(label_shape);
        }

        this->item_index_ = 0;
    }

    template <typename Dtype>
    void DragonMmapDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

        Dtype *data_ptr = this->data_ + this->item_index_ * this->num_elements_per_data_item_();
        char *ptr = (char *)data_ptr;
        /* Populate the data */
        size_t block_size = (size_t)1 << 21;
        size_t data_batch_size = batchsize * this->size_per_data_item_();
        std::stringstream ss;
        for (size_t size_index = 0; size_index < data_batch_size; size_index += block_size)
        {
            char tmp = ptr[size_index];
            ss << tmp;
        }
        
        top[0]->set_cpu_data(data_ptr);

        if (this->output_labels_)
        {
            vector<int> label_shape(1, batchsize);
            top[1]->Reshape(label_shape);

            Dtype *labels_ptr = this->labels_ + this->item_index_;
            ptr = (char *)labels_ptr;

            size_t labels_batch_size = batchsize * sizeof(Dtype);
            for (size_t size_index = 0; size_index < labels_batch_size; size_index += block_size)
            {
                char tmp = labels_ptr[size_index];
                ss << tmp;
            }

            top[1]->set_cpu_data(labels_ptr);
        }

        this->item_index_ += batchsize;
        if (this->item_index_ >= this->data_length_)
            this->item_index_ = 0;
    }

    INSTANTIATE_CLASS(DragonMmapDataLayer);
    REGISTER_LAYER_CLASS(DragonMmapData);

}  // namespace caffe

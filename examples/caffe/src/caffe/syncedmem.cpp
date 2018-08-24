#include <dragon.h>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <mutex>

#include <gflags/gflags.h>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

DEFINE_bool(enable_dragon, false,
    "Optional; enable DRAGON as memory backend.");
DEFINE_string(dragon_tmp_folder, "/tmp",
    "Optional; the folder to hold the memory-backend files used by DRAGON.");
DEFINE_uint64(dragon_enable_threshold, (size_t)1 << 21,
    "Optional; enable DRAGON on the memory regions that are larger than or equal to this threshold.");

DEFINE_bool(enable_uvm, false,
    "Optional; enable UVM as memory backend.");

DEFINE_bool(enable_mmap, false,
    "Optional; enable mmap as memory backend.");

static long _memfile_number_ = 0;
static const size_t _memfile_totalsize = (size_t)1 << 33;
static size_t _memfile_current_usage = 0;
static void *_memfile_current_ptr = NULL;
static std::mutex _memfile_mutex;
//static size_t _memfile_total_usage = 0;

namespace caffe {
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
    dragon_ptr_(NULL), use_dragon_(false), use_mmap_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
  CHECK(!(FLAGS_enable_dragon && FLAGS_enable_uvm && FLAGS_enable_mmap)) << "DRAGON, UVM, and mmap memory backend options cannot be activated at the same time";
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
    dragon_ptr_(NULL), use_dragon_(false), use_mmap_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
  CHECK(!(FLAGS_enable_dragon && FLAGS_enable_uvm && FLAGS_enable_mmap)) << "DRAGON, UVM, and mmap memory backend options cannot be activated at the same time";
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_ && !use_mmap_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  std::stringstream sstm;
  check_device();
  if (use_dragon_ && head_ != UNINITIALIZED)
    head_ = SYNCED;
  switch (head_) {
  case UNINITIALIZED:
    if (FLAGS_enable_dragon && size_ >= FLAGS_dragon_enable_threshold) {
        _memfile_mutex.lock();
        if (_memfile_current_ptr == NULL || size_ + _memfile_current_usage > _memfile_totalsize) {
            sstm << FLAGS_dragon_tmp_folder << "tmp_" << _memfile_number_ << ".mem";
            if (dragon_map(sstm.str().c_str(), _memfile_totalsize, D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE, &_memfile_current_ptr) != D_OK) {
                std::cerr << "Cannot dragon_map " << sstm.str() << std::endl;
                abort();
            }
            _memfile_current_usage = 0;
            ++_memfile_number_;
        }
        dragon_ptr_ = (void *)((char *)_memfile_current_ptr + _memfile_current_usage);
        _memfile_current_usage += size_;
        //_memfile_total_usage += size_;
        //LOG(INFO) << "_memfile_total_usage: " << _memfile_total_usage;
        head_ = SYNCED;
        use_dragon_ = true;
        own_cpu_data_ = false;
        own_gpu_data_ = false;
        _memfile_mutex.unlock();
    }
#ifndef CPU_ONLY
    else if (FLAGS_enable_uvm && size_ >= FLAGS_dragon_enable_threshold) {
        CUDA_CHECK(cudaMallocManaged(&dragon_ptr_, size_));
        head_ = SYNCED;
        use_dragon_ = true;
        own_cpu_data_ = false;
        own_gpu_data_ = false;
    }
#endif
    else if (FLAGS_enable_mmap && size_ >= FLAGS_dragon_enable_threshold) {
        _memfile_mutex.lock();
        if (_memfile_current_ptr == NULL || size_ + _memfile_current_usage > _memfile_totalsize) {
            int fd;
            sstm << FLAGS_dragon_tmp_folder << "tmp_" << _memfile_number_ << ".mem";
            if ((fd = creat(sstm.str().c_str(), S_IRUSR | S_IWUSR)) >= 0)
                close(fd);
            fd = open(sstm.str().c_str(), O_RDWR | O_LARGEFILE);
            CHECK(fd >= 0) << "Cannot open file " << sstm.str();
            CHECK(ftruncate(fd, _memfile_totalsize) == 0) << "Cannot truncate file " << sstm.str();
            _memfile_current_ptr = mmap(NULL, _memfile_totalsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd, 0);
            CHECK(dragon_ptr_ != MAP_FAILED) << "Cannot mmap file " << sstm.str();
            _memfile_current_usage = 0;
            ++_memfile_number_;
        }

        cpu_ptr_ = (void *)((char *)_memfile_current_ptr + _memfile_current_usage);
        _memfile_current_usage += size_;

        head_ = HEAD_AT_CPU;
        use_mmap_ = true;
        own_cpu_data_ = true;
        _memfile_mutex.unlock();
    }
    else {
        CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
        caffe_memset(size_, 0, cpu_ptr_);
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = true;
        use_dragon_ = false;
    }
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  std::stringstream sstm;
  check_device();
#ifndef CPU_ONLY
  if (use_dragon_ && head_ != UNINITIALIZED)
    head_ = SYNCED;
  switch (head_) {
  case UNINITIALIZED:
    if (FLAGS_enable_dragon && size_ >= FLAGS_dragon_enable_threshold) {
        _memfile_mutex.lock();
        if (_memfile_current_ptr == NULL || size_ + _memfile_current_usage > _memfile_totalsize) {
            sstm << FLAGS_dragon_tmp_folder << "tmp_" << _memfile_number_ << ".mem";
            if (dragon_map(sstm.str().c_str(), _memfile_totalsize, D_F_READ | D_F_WRITE | D_F_CREATE | D_F_VOLATILE, &_memfile_current_ptr) != D_OK) {
                std::cerr << "Cannot dragon_map " << sstm.str() << std::endl;
                abort();
            }
            _memfile_current_usage = 0;
            ++_memfile_number_;
        }
        dragon_ptr_ = (void *)((char *)_memfile_current_ptr + _memfile_current_usage);
        _memfile_current_usage += size_;
        //_memfile_total_usage += size_;
        //LOG(INFO) << "_memfile_total_usage: " << _memfile_total_usage;
        head_ = SYNCED;
        use_dragon_ = true;
        own_cpu_data_ = false;
        own_gpu_data_ = false;
        _memfile_mutex.unlock();
    }
    else if (FLAGS_enable_uvm && size_ >= FLAGS_dragon_enable_threshold) {
        CUDA_CHECK(cudaMallocManaged(&dragon_ptr_, size_));
        head_ = SYNCED;
        use_dragon_ = true;
        own_cpu_data_ = false;
        own_gpu_data_ = false;
    }
    else {
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
        caffe_gpu_memset(size_, 0, gpu_ptr_);
        head_ = HEAD_AT_GPU;
        own_gpu_data_ = true;
        use_dragon_ = false;
    }
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  if (use_dragon_)
    return (const void*)dragon_ptr_;
  else
    return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_ && !use_mmap_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
  use_dragon_ = false;
  use_mmap_ = false;
}

void SyncedMemory::set_dragon_data(void* data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_ && !use_mmap_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = NULL;
  dragon_ptr_ = data;
  head_ = SYNCED;
  own_cpu_data_ = false;
  use_dragon_ = true;
  use_mmap_ = false;
}

const void* SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  if (use_dragon_)
    return (const void*)dragon_ptr_;
  else
    return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
  use_dragon_ = false;
  use_mmap_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  if (use_dragon_) {
      head_ = SYNCED;
      return dragon_ptr_;
  }
  else {
      head_ = HEAD_AT_CPU;
      return cpu_ptr_;
  }
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  if (use_dragon_) {
      head_ = SYNCED;
      return dragon_ptr_;
  }
  else {
      head_ = HEAD_AT_GPU;
      return gpu_ptr_;
  }
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  if (!use_dragon_) {
      CHECK(head_ == HEAD_AT_CPU);
      if (gpu_ptr_ == NULL) {
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
        own_gpu_data_ = true;
      }
      const cudaMemcpyKind put = cudaMemcpyHostToDevice;
      CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  }
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

}  // namespace caffe


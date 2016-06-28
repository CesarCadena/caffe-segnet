#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SparseEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  for (int i = 0; i < bottom[0]->count(); ++i) {
      if (!bottom[2]->cpu_data()[i]) {  // missed label
	 diff_.mutable_cpu_data()[i] = Dtype(0.0);		  
      }
  }  
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SparseEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
     const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
     caffe_gpu_axpby(
        bottom[0]->count(),              // count
        alpha,                              // alpha
        diff_.gpu_data(),                   // a
        Dtype(0),                           // beta
        bottom[0]->mutable_gpu_diff());  // b
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseEuclideanLossLayer);

}  // namespace caffe

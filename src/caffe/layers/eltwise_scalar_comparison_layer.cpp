#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"


namespace caffe {

template <typename Dtype>
void EltwiseScalarComparisonLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);  
  value_.clear();
  std::copy(this->layer_param_.eltwise_scalarcomparison_param().value().begin(),
      this->layer_param_.eltwise_scalarcomparison_param().value().end(),
      std::back_inserter(value_));
}

template <typename Dtype>
void EltwiseScalarComparisonLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
	top_data[i] = Dtype(0);
	for (int j = 0; j < value_.size(); ++j) {
		if (bottom_data[i] == value_[j]) {
			top_data[i] =  Dtype(1);
		}
    }
    
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(EltwiseScalarComparisonLayer, Forward);
#endif

INSTANTIATE_CLASS(EltwiseScalarComparisonLayer);
REGISTER_LAYER_CLASS(EltwiseScalarComparison);

}  // namespace caffe

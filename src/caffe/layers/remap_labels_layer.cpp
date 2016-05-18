#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"


namespace caffe {

template <typename Dtype>
void RemapLabelsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);  
  oldlabel_.clear();
  std::copy(this->layer_param_.remap_labels_param().oldlabel().begin(),
      this->layer_param_.remap_labels_param().oldlabel().end(),
      std::back_inserter(oldlabel_));
  newlabel_.clear();
  std::copy(this->layer_param_.remap_labels_param().newlabel().begin(),
      this->layer_param_.remap_labels_param().newlabel().end(),
      std::back_inserter(newlabel_));
}

template <typename Dtype>
void RemapLabelsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
	top_data[i] = Dtype(0);// bottom_data[i];
	for (int j = 0; j < oldlabel_.size(); ++j) {
		if (bottom_data[i] == oldlabel_[j]) {
			top_data[i] =  newlabel_[j];
		}
    }
    
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(RemapLabelsLayer, Forward);
#endif

INSTANTIATE_CLASS(RemapLabelsLayer);
REGISTER_LAYER_CLASS(RemapLabels);

}  // namespace caffe

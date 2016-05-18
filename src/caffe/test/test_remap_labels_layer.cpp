#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/neuron_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class RemapLabelsLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  RemapLabelsLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {    
    // fill the values
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      blob_bottom_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~RemapLabelsLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RemapLabelsLayerTest, TestDtypesAndDevices);


TYPED_TEST(RemapLabelsLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RemapLabelsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(RemapLabelsLayerTest, Test) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_remap_labels_param()->add_oldlabel(2.);
  layer_param.mutable_remap_labels_param()->add_newlabel(1.);
  RemapLabelsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();  
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
	if (bottom_data[i] == 2.) {
		EXPECT_NE(top_data[i], 2.);    
	}
  }
}

TYPED_TEST(RemapLabelsLayerTest, Test2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_remap_labels_param()->add_oldlabel(2.);
  layer_param.mutable_remap_labels_param()->add_oldlabel(3.);
  layer_param.mutable_remap_labels_param()->add_newlabel(1.);
  layer_param.mutable_remap_labels_param()->add_newlabel(1.);
  RemapLabelsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();    
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NE(top_data[i], 2.);
    EXPECT_NE(top_data[i], 3.);
    if (bottom_data[i] == 2.) {
      EXPECT_EQ(top_data[i], 1.);
    }
    if (bottom_data[i] == 3.) {
      EXPECT_EQ(top_data[i], 1.);
    }
  }
}

TYPED_TEST(RemapLabelsLayerTest, Test3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_remap_labels_param()->add_oldlabel(4.);
  layer_param.mutable_remap_labels_param()->add_oldlabel(3.);
  layer_param.mutable_remap_labels_param()->add_newlabel(1.);
  layer_param.mutable_remap_labels_param()->add_newlabel(2.);
  layer_param.mutable_remap_labels_param()->add_oldlabel(1.);
  layer_param.mutable_remap_labels_param()->add_oldlabel(2.);
  layer_param.mutable_remap_labels_param()->add_newlabel(3.);
  layer_param.mutable_remap_labels_param()->add_newlabel(4.);
  RemapLabelsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();    
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {    
    if (bottom_data[i] == 1.) {
      EXPECT_EQ(top_data[i], 3.);
    }
    if (bottom_data[i] == 2.) {
      EXPECT_EQ(top_data[i], 4.);
    }
    if (bottom_data[i] == 3.) {
      EXPECT_EQ(top_data[i], 2.);
    }
    if (bottom_data[i] == 4.) {
      EXPECT_EQ(top_data[i], 1.);
    }
  }
}

}  // namespace caffe

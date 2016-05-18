#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/neuron_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class EltwiseScalarComparisonLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  EltwiseScalarComparisonLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {    
    // fill the values
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      blob_bottom_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~EltwiseScalarComparisonLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EltwiseScalarComparisonLayerTest, TestDtypesAndDevices);


TYPED_TEST(EltwiseScalarComparisonLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseScalarComparisonLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(EltwiseScalarComparisonLayerTest, Test) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_eltwise_scalarcomparison_param()->add_value(2.);
  EltwiseScalarComparisonLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();  
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_LE(top_data[i], 1.);
    if (top_data[i] == 0) {
      EXPECT_NE(bottom_data[i], 2.);
    }
    if (top_data[i] == 1) {
      EXPECT_EQ(bottom_data[i], 2.);
    }
  }
}

TYPED_TEST(EltwiseScalarComparisonLayerTest, Test2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kValue0 = 2.;
  const Dtype kValue1 = 3.;
  layer_param.mutable_eltwise_scalarcomparison_param()->add_value(kValue0);
  layer_param.mutable_eltwise_scalarcomparison_param()->add_value(kValue1);  
  EltwiseScalarComparisonLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();    
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_LE(top_data[i], 1.);
    if (top_data[i] == 0) {
      EXPECT_NE(bottom_data[i], kValue0);
      EXPECT_NE(bottom_data[i], kValue1);
    }
  }
}

}  // namespace caffe

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SparseEuclideanLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SparseEuclideanLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_mask_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param0, filler_param1;
    filler_param0.set_min(0.1);
    filler_param0.set_max(1.0);
    UniformFiller<Dtype>  filler0(filler_param0);
    //GaussianFiller<Dtype> filler0(filler_param0);
    filler0.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler0.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    filler_param1.set_value(1.0);    
    ConstantFiller<Dtype>  filler1(filler_param1);
    filler1.Fill(this->blob_bottom_mask_);
    for (int i = 0; i < blob_bottom_mask_->count(); ++i) {
		if (!(caffe_rng_rand() % 2)){
			blob_bottom_mask_->mutable_cpu_data()[i] = Dtype(0.0);
		}
    }    
    blob_bottom_vec_.push_back(blob_bottom_mask_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SparseEuclideanLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_mask_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    SparseEuclideanLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    //cout << "count: " << blob_bottom_data_->count() << "  num: "<<blob_bottom_data_->num()<< endl;
	// manually compute
	int num = this->blob_bottom_data_->num();
	Dtype loss(0);
	for (int i = 0; i < this->blob_bottom_data_->count(); ++i) {
		if (this->blob_bottom_mask_->cpu_data()[i] > Dtype(0.0)){
			Dtype diff = this->blob_bottom_data_->cpu_data()[i] -
			             this->blob_bottom_label_->cpu_data()[i];
			loss += diff*diff;
	    }
	}
	Dtype Eloss(0);
	for (int i = 0; i < this->blob_bottom_data_->count(); ++i) {
		Dtype diff = this->blob_bottom_data_->cpu_data()[i] -
		             this->blob_bottom_label_->cpu_data()[i];
		Eloss += diff*diff;
	}		
	loss /= static_cast<Dtype>(num) * Dtype(2);
	Eloss /= static_cast<Dtype>(num) * Dtype(2);
	cout << "EuclideanLoss: " << Eloss << "  SparseEuclideanLoss: "<<loss<< endl;
	EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);	
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_mask_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SparseEuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SparseEuclideanLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(SparseEuclideanLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SparseEuclideanLossLayer<Dtype> layer(layer_param);  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);  
}

}  // namespace caffe

 // --------------------------------------------------------
 // Proposal Layer C++ Implement
 // Copyright (c) 2017 Lenovo
 // Written by Zou Jinyi
 // --------------------------------------------------------
#include <vector>

#include "caffe/layers/proposal_layer.hpp"

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ProposalParameter& param = this->layer_param_.proposal_param();
  
  feat_stride_ = param.feat_stride();
  anchor_base_size_ = param.anchor_base_size();
  if (param.anchor_scale() == 3)
  {
	  anchor_scale_.push_back(8.0);
	  anchor_scale_.push_back(16.0);
	  anchor_scale_.push_back(32.0);
  }
  else
  {
	  anchor_scale_.push_back(32.0);
  }
  if (param.anchor_ratio() == 3)
  {
	  anchor_ratio_.push_back(0.5);
	  anchor_ratio_.push_back(1.0);
	  anchor_ratio_.push_back(2.0);
  }
  else
  {
	  anchor_ratio_.push_back(1.0);
  }
  max_rois_ = param.max_rois();
  Generate_anchors();
}

template <typename Dtype>	  
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* score = bottom[0]->cpu_data();
	const Dtype* bbox_deltas = bottom[1]->cpu_data();
	const Dtype* im_info = bottom[2]->cpu_data();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	float thresh = 0.3;
	vector<vector<float> > select_anchor;
	vector<float> confidence;
	vector<vector<float> > bbox;
	int anchor_num = anchor_scale_.size()*anchor_ratio_.size();
	
	for (int k = 0; k < anchor_num; k++)
	{
		float w = anchor_boxes_[4 * k + 2] - anchor_boxes_[4 * k] + 1;
		float h = anchor_boxes_[4 * k + 3] - anchor_boxes_[4 * k + 1] + 1;
		float x_ctr = anchor_boxes_[4 * k] + 0.5 * (w - 1);
		float y_ctr = anchor_boxes_[4 * k + 1] + 0.5 * (h - 1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (score[anchor_num*height*width + (k * height + i) * width + j] >= thresh)
				{
					vector<float> tmp_anchor;
					vector<float> tmp_bbox;

					tmp_anchor.push_back(j * feat_stride_+ x_ctr);
					tmp_anchor.push_back(i * feat_stride_+ y_ctr);
					tmp_anchor.push_back(w);
					tmp_anchor.push_back(h);
					select_anchor.push_back(tmp_anchor);
					confidence.push_back(score[anchor_num*height*width + (k * height + i) * width + j]);
					tmp_bbox.push_back(bbox_deltas[(4 * k * height + i) * width + j]);
					tmp_bbox.push_back(bbox_deltas[((4 * k +1) * height + i) * width + j]);
					tmp_bbox.push_back(bbox_deltas[((4 * k + 2) * height + i) * width + j]);
					tmp_bbox.push_back(bbox_deltas[((4 * k + 3) * height + i) * width + j]);
					bbox.push_back(tmp_bbox);
				}
			}
		}
	}
	vector<vector<float> > pred_boxes;
	bbox_transform_inv(im_info[1], im_info[0], bbox, select_anchor, pred_boxes);
	
    apply_nms(pred_boxes, confidence);

	int num = pred_boxes.size() > max_rois_ ? max_rois_ : pred_boxes.size();

	vector<int> proposal_shape;
	proposal_shape.push_back(num);
	proposal_shape.push_back(5);
	top[0]->Reshape(proposal_shape);
	Dtype* top_data = top[0]->mutable_cpu_data();
	for (int i = 0; i < num; i++)
	{
		top_data[5 * i] = 0;
		top_data[5 * i + 1] = pred_boxes[i][0];
		top_data[5 * i + 2] = pred_boxes[i][1];
		top_data[5 * i + 3] = pred_boxes[i][2];
		top_data[5 * i + 4] = pred_boxes[i][3];
	}
}


//generate anchors
template <typename Dtype>
void ProposalLayer<Dtype>::Generate_anchors() {
	vector<float> base_anchor;
	base_anchor.push_back(0);
	base_anchor.push_back(0);
	base_anchor.push_back(anchor_base_size_ - 1);
	base_anchor.push_back(anchor_base_size_ - 1);
	vector<float> anchors_ratio;
	_ratio_enum(base_anchor, anchors_ratio);
	_scale_enum(anchors_ratio, anchor_boxes_);
}

template <typename Dtype>
void ProposalLayer<Dtype>::_whctrs(vector <float> anchor, vector<float> &ctrs) {
	float w = anchor[2] - anchor[0] + 1;
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);
	ctrs.push_back(w);
	ctrs.push_back(h);
	ctrs.push_back(x_ctr);
	ctrs.push_back(y_ctr);
}

template <typename Dtype>
void ProposalLayer<Dtype>::_ratio_enum(vector<float> anchor, vector<float> &anchors_ratio) {
	vector<float> ctrs;
	_whctrs(anchor, ctrs);
	float size = ctrs[0] * ctrs[1];
	int ratio_num = anchor_ratio_.size();
	for (int i = 0; i < ratio_num; i++)
	{
		float ratio = size / anchor_ratio_[i];
		int ws = int(round(sqrt(ratio)));
		int hs = int(round(ws * anchor_ratio_[i]));
		vector<float> ctrs_in;
		ctrs_in.push_back(ws);
		ctrs_in.push_back(hs);
		ctrs_in.push_back(ctrs[2]);
		ctrs_in.push_back(ctrs[3]);
		_mkanchors(ctrs_in, anchors_ratio);
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::_scale_enum(vector<float> anchors_ratio, vector<float> &anchor_boxes) {
	int anchors_ratio_num = anchors_ratio.size() / 4;
	for (int i = 0; i < anchors_ratio_num; i++)
	{
		vector<float> anchor;
		anchor.push_back(anchors_ratio[i * 4]);
		anchor.push_back(anchors_ratio[i * 4 + 1]);
		anchor.push_back(anchors_ratio[i * 4 + 2]);
		anchor.push_back(anchors_ratio[i * 4 + 3]);
		vector<float> ctrs;
		_whctrs(anchor, ctrs);
		int scale_num = anchor_scale_.size();
		for (int j = 0; j < scale_num; j++)
		{
			float ws = ctrs[0] * anchor_scale_[j];
			float hs = ctrs[1] * anchor_scale_[j];
			vector<float> ctrs_in;
			ctrs_in.push_back(ws);
			ctrs_in.push_back(hs);
			ctrs_in.push_back(ctrs[2]);
			ctrs_in.push_back(ctrs[3]);
			_mkanchors(ctrs_in, anchor_boxes_);
		}
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::_mkanchors(vector<float> ctrs, vector<float> &anchors) {
	anchors.push_back(ctrs[2] - 0.5*(ctrs[0] - 1));
	anchors.push_back(ctrs[3] - 0.5*(ctrs[1] - 1));
	anchors.push_back(ctrs[2] + 0.5*(ctrs[0] - 1));
	anchors.push_back(ctrs[3] + 0.5*(ctrs[1] - 1));
}

template <typename Dtype>
void ProposalLayer<Dtype>::bbox_transform_inv(int img_width, int img_height, vector<vector<float> > bbox, vector<vector<float> > select_anchor, vector<vector<float> > &pred)
{
	int num = bbox.size();
	for (int i = 0; i< num; i++)
	{
			float dx = bbox[i][0];
			float dy = bbox[i][1];
			float dw = bbox[i][2];
			float dh = bbox[i][3];
			float pred_ctr_x = select_anchor[i][0] + select_anchor[i][2]*dx;
			float pred_ctr_y = select_anchor[i][1] + select_anchor[i][3] *dy;
			float pred_w = select_anchor[i][2] * exp(dw);
			float pred_h = select_anchor[i][3] * exp(dh);
			vector<float> tmp_pred;
			tmp_pred.push_back(max(min(pred_ctr_x - 0.5* pred_w, img_width - 1), 0));
			tmp_pred.push_back(max(min(pred_ctr_y - 0.5* pred_h, img_height - 1), 0));
			tmp_pred.push_back(max(min(pred_ctr_x + 0.5* pred_w, img_width - 1), 0));
			tmp_pred.push_back(max(min(pred_ctr_y + 0.5* pred_h, img_height - 1), 0));
			pred.push_back(tmp_pred);
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::apply_nms(vector<vector<float> > &pred_boxes, vector<float> &confidence)
{
	for (int i = 0; i < pred_boxes.size()-1; i++)
	{
		float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1) *(pred_boxes[i][3] - pred_boxes[i][1] + 1);
		for (int j = i + 1; j < pred_boxes.size(); j++)
		{
			float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1) *(pred_boxes[j][3] - pred_boxes[j][1] + 1);

			float x1 = max(pred_boxes[i][0], pred_boxes[j][0]);
			float y1 = max(pred_boxes[i][1], pred_boxes[j][1]);
			float x2 = min(pred_boxes[i][2], pred_boxes[j][2]);
			float y2 = min(pred_boxes[i][3], pred_boxes[j][3]);

			float width = x2 - x1;
			float height = y2 - y1;
			if (width > 0 && height > 0)
			{
				float IOU = width * height / (s1 + s2 - width * height);
				if (IOU > 0.7)
				{
					if (confidence[i] >= confidence[j])
					{
						pred_boxes.erase(pred_boxes.begin() + j);
						confidence.erase(confidence.begin() + j);
						j--;
					}
					else
					{
						pred_boxes.erase(pred_boxes.begin() + i);
						confidence.erase(confidence.begin() + i);
						i--;
						break;
					}
				}
			}
		}
	}
}


INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))

//background and car
const int CLASS_NUM = 21;

float CONF_THRESH = 0.7;
float NMS_THRESH = 0.5;

/*
* ===  Class  ======================================================================
*         Name:  Detector
*  Description:  FasterRCNN CXX Detector
* =====================================================================================
*/
class Detector {
public:
	Detector(const string& model_file, const string& weights_file);
	void Detect(string im_name);
    void Detect_video(string im_name);
	void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
	void vis_detections(cv::Mat image, vector<vector<float> > pred_boxes, vector<float> confidence, float CONF_THRESH);
	void boxes_sort(int num, const float* pred, float* sorted_pred);
	void apply_nms(vector<vector<float> > &pred_boxes, vector<float> &confidence);

private:
	boost::shared_ptr<Net<float> > net_;
	Detector() {}
};

struct Info
{
	float score;
	const float* head = NULL;
};


using namespace caffe;
using namespace std;

/*
* ===  FUNCTION  ======================================================================
*         Name:  Detector
*  Description:  Load the model file and weights file
* =====================================================================================
*/
//load modelfile and weights
Detector::Detector(const string& model_file, const string& weights_file)
{
	net_ = boost::shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(weights_file);
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  Detect
*  Description:  Perform detection operation
*                 Warning the max input size should less than 1000*600
* =====================================================================================
*/
bool compare_score(const Info& Info1, const Info& Info2)
{
	return Info1.score > Info2.score;
}
//perform detection operation
//input image max size 1000*600
void Detector::Detect(string im_name)
{
	const int  max_input_side = 500;
	const int  min_input_side = 300;

	cv::Mat cv_img = cv::imread(im_name);
	cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0, 0, 0));
	if (cv_img.empty())
	{
		std::cout << "Can not get the image file !" << endl;
		return;
	}
	int max_side = max(cv_img.rows, cv_img.cols);
	int min_side = min(cv_img.rows, cv_img.cols);

	float max_side_scale = float(max_side) / float(max_input_side);
	float min_side_scale = float(min_side) / float(min_input_side);
	float max_scale = max(max_side_scale, min_side_scale);

	float img_scale = 1;

	if (max_scale < 1)
	{
		img_scale = float(1) / max_scale;
	}

	int height = int(cv_img.rows * img_scale);
	int width = int(cv_img.cols * img_scale);
	cv::Mat cv_resized;

	std::cout << "imagename " << im_name << endl;
	float im_info[3];
	float *data_buf= new float[height * width * 3];
	float *boxes = NULL;
	float *pred = NULL;
	float *pred_per_class = NULL;
	float *sorted_pred_cls = NULL;
	int *keep = NULL;
	const float* bbox_delt = NULL;
	const float* rois = NULL;
	const float* pred_cls = NULL;
	int num;

	for (int h = 0; h < cv_img.rows; ++h)
	{
		for (int w = 0; w < cv_img.cols; ++w)
		{
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(102.9801);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(115.9465);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(122.7717);

		}
	}

	cv::resize(cv_new, cv_resized, cv::Size(width, height));
	im_info[0] = cv_resized.rows;
	im_info[1] = cv_resized.cols;
	im_info[2] = img_scale;


	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			data_buf[(0 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
			data_buf[(1 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
			data_buf[(2 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}

	net_->blob_by_name("data")->Reshape(1, 3, height, width);
	net_->blob_by_name("data")->set_cpu_data(data_buf);
	net_->blob_by_name("im_info")->set_cpu_data(im_info);
    clock_t t1 = clock();
	net_->ForwardFrom(0);
	std::cout << " Time Using GPU-CUDNN: " << (clock() - t1)*1.0 / CLOCKS_PER_SEC << std::endl;
	bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	num = net_->blob_by_name("rois")->num();

	rois = net_->blob_by_name("rois")->cpu_data();
	pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
	boxes = new float[num * 4];
	pred = new float[num * 5 * CLASS_NUM];
	pred_per_class = new float[num * 5];
	sorted_pred_cls = new float[num * 5];
	keep = new int[num];

	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < 4; c++)
		{
			boxes[n * 4 + c] = rois[n * 5 + c + 1] / img_scale;
		}
	}

	bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, cv_img.rows, cv_img.cols);
	for (int i = 1; i < CLASS_NUM; i++)
	{
		for (int j = 0; j< num; j++)
		{
			for (int k = 0; k < 5; k++)
			{
				pred_per_class[j * 5 + k] = pred[(i*num + j) * 5 + k];
			}
		}

		vector<vector<float> > pred_boxes;
		vector<float> confidence;
		for (int j = 0; j < num; j++)
		{
			vector<float> tmp_box;
			tmp_box.push_back(pred_per_class[j * 5 + 0]);
			tmp_box.push_back(pred_per_class[j * 5 + 1]);
			tmp_box.push_back(pred_per_class[j * 5 + 2]);
			tmp_box.push_back(pred_per_class[j * 5 + 3]);
			pred_boxes.push_back(tmp_box);
			confidence.push_back(pred_per_class[j * 5 + 4]);
		}



		apply_nms(pred_boxes, confidence);
		vis_detections(cv_img, pred_boxes, confidence, CONF_THRESH);
	}

	cv::imwrite("vis.jpg", cv_img);
	delete[]boxes;
	delete[]pred;
	delete[]pred_per_class;
	delete[]keep;
	delete[]sorted_pred_cls;
    delete[]data_buf;

}

void Detector::Detect_video(string video_source)
{
    std::string outFlie = "result.avi";
    cv::VideoCapture cap_;
    cap_.open(video_source);
    if(!cap_.isOpened())
    {
        cout<<"error opened video"<<endl;
    }
    else
    {
        float fps = cap_.get(CV_CAP_PROP_FPS);
        //float fourcc = cap_.get(CV_CAP_PROP_FOURCC);

	    const int  max_input_side = 500;
	    const int  min_input_side = 300;
        cv::Mat cv_img;
        cap_ >> cv_img;

	    int max_side = max(cv_img.rows, cv_img.cols);
	    int min_side = min(cv_img.rows, cv_img.cols);
        
	    float max_side_scale = float(max_side) / float(max_input_side);
	    float min_side_scale = float(min_side) / float(min_input_side);
	    float max_scale = max(max_side_scale, min_side_scale);

	    float img_scale = 1;

	    if (max_scale > 1)
	    {
		    img_scale = float(1) / max_scale;
	    }

	    int height = int(cv_img.rows * img_scale);
	    int width = int(cv_img.cols * img_scale);
        //cv::Size shape(cv_img.cols,cv_img.rows);

        //cv::VideoWriter write(outFlie, fourcc,fps,shape);
        cv::VideoWriter write(outFlie, CV_FOURCC('M', 'J', 'P', 'G'),fps,cv::Size(cv_img.cols,cv_img.rows));
	    
        cv::Mat cv_resized;

	    float im_info[3];
	    float *data_buf= new float[height * width * 3];
	    int num;
	    cv::Mat cv_new(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
	
        while (true)
	    {
	        float *boxes = NULL;
	        float *pred = NULL;
    	    float *pred_per_class = NULL;
	        float *sorted_pred_cls = NULL;
    	    int *keep = NULL;
	        const float* bbox_delt = NULL;
    	    const float* rois = NULL;
	        const float* pred_cls = NULL;
	        cv::resize(cv_img, cv_resized, cv::Size(width, height));
	        for (int h = 0; h < cv_resized.rows; ++h)
	        {
		        for (int w = 0; w < cv_resized.cols; ++w)
		        {
			        cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(102.9801);
			        cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(115.9465);
			        cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(122.7717);
		        }   
	        }

	        im_info[0] = cv_new.rows;
    	    im_info[1] = cv_new.cols;
	        im_info[2] = img_scale;


    	    for (int h = 0; h < height; ++h)
	        {
		        for (int w = 0; w < width; ++w)
		        {
    			data_buf[(0 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[0]);
	    		data_buf[(1 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[1]);
		    	data_buf[(2 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[2]);
		        }
	        }

    	    net_->blob_by_name("data")->Reshape(1, 3, height, width);
	        net_->blob_by_name("data")->set_cpu_data(data_buf);
	        net_->blob_by_name("im_info")->set_cpu_data(im_info);
            clock_t t1 = clock();
	        net_->ForwardFrom(0);
	        std::cout << " Forward Time Using GPU-CUDNN: " << (clock() - t1)*1.0 / CLOCKS_PER_SEC << std::endl;
    	    bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	        num = net_->blob_by_name("rois")->num();

	        rois = net_->blob_by_name("rois")->cpu_data();
    	    pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
	        boxes = new float[num * 4];
	        pred = new float[num * 5 * CLASS_NUM];
    	    pred_per_class = new float[num * 5];
	        sorted_pred_cls = new float[num * 5];
	        keep = new int[num];

	        for (int n = 0; n < num; n++)
	        {
    		    for (int c = 0; c < 4; c++)
	    	    {
		    	    boxes[n * 4 + c] = rois[n * 5 + c + 1] / img_scale;
		        }
	        }

    	    bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, cv_img.rows, cv_img.cols);
	    
            for (int i = 1; i < CLASS_NUM; i++)
	        {
		        for (int j = 0; j< num; j++)
		        {
			        for (int k = 0; k < 5; k++)
			        {
				        pred_per_class[j * 5 + k] = pred[(i*num + j) * 5 + k];
			        }
		        }

    		    vector<vector<float> > pred_boxes;
	    	    vector<float> confidence;
		        for (int j = 0; j < num; j++)
		        {
    			    vector<float> tmp_box;
	    		    tmp_box.push_back(pred_per_class[j * 5 + 0]);
		    	    tmp_box.push_back(pred_per_class[j * 5 + 1]);
			        tmp_box.push_back(pred_per_class[j * 5 + 2]);
        			tmp_box.push_back(pred_per_class[j * 5 + 3]);
	        		pred_boxes.push_back(tmp_box);
		        	confidence.push_back(pred_per_class[j * 5 + 4]);
		        }

    		    apply_nms(pred_boxes, confidence);
	    	    vis_detections(cv_img, pred_boxes, confidence, CONF_THRESH);
            }

	        delete[]boxes;
	        delete[]pred;
    	    delete[]pred_per_class;
	        delete[]keep;
    	    delete[]sorted_pred_cls;
            write<<cv_img;
	        cap_>>cv_img;
            if (cv_img.empty())
            {
                break;
            }
        }
        cap_.release();
        write.release();
        delete[]data_buf;
    }
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  vis_detections
*  Description:  Visuallize the detection result
* =====================================================================================
*/
void Detector::vis_detections(cv::Mat image, vector<vector<float> > pred_boxes, vector<float> confidence, float CONF_THRESH)
{
	for(int i=0; i<confidence.size();i++)
	{
		if (confidence[i] > CONF_THRESH)
		{
			cv::rectangle(image, cv::Point(pred_boxes[i][0], pred_boxes[i][1]), cv::Point(pred_boxes[i][2], pred_boxes[i][3]), cv::Scalar(255, 0, 0));

		}
	}
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  boxes_sort
*  Description:  Sort the bounding box according score
* =====================================================================================
*/
//Using for box sort


void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred)
{
	vector<Info> my;
	Info tmp;
	for (int i = 0; i< num; i++)
	{
		tmp.score = pred[i * 5 + 4];
		tmp.head = pred + i * 5;
		my.push_back(tmp);
	}
	std::sort(my.begin(), my.end(), compare_score);
	for (int i = 0; i<num; i++)
	{
		for (int j = 0; j<5; j++)
			sorted_pred[i * 5 + j] = my[i].head[j];
	}
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  bbox_transform_inv
*  Description:  Compute bounding box regression value
* =====================================================================================
*/
void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width)
{
	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	for (int i = 0; i< num; i++)
	{
		width = boxes[i * 4 + 2] - boxes[i * 4 + 0] + 1.0;
		height = boxes[i * 4 + 3] - boxes[i * 4 + 1] + 1.0;
		ctr_x = boxes[i * 4 + 0] + 0.5 * width;
		ctr_y = boxes[i * 4 + 1] + 0.5 * height;
		for (int j = 1; j< CLASS_NUM; j++)
		{
			dx = box_deltas[(i*CLASS_NUM + j) * 4 + 0];
			dy = box_deltas[(i*CLASS_NUM + j) * 4 + 1];
			dw = box_deltas[(i*CLASS_NUM + j) * 4 + 2];
			dh = box_deltas[(i*CLASS_NUM + j) * 4 + 3];
			pred_ctr_x = ctr_x + width*dx;
			pred_ctr_y = ctr_y + height*dy;
			pred_w = width * exp(dw);
			pred_h = height * exp(dh);
			pred[(j*num + i) * 5 + 0] = max(min(pred_ctr_x - 0.5* pred_w, img_width - 1), 0);
			pred[(j*num + i) * 5 + 1] = max(min(pred_ctr_y - 0.5* pred_h, img_height - 1), 0);
			pred[(j*num + i) * 5 + 2] = max(min(pred_ctr_x + 0.5* pred_w, img_width - 1), 0);
			pred[(j*num + i) * 5 + 3] = max(min(pred_ctr_y + 0.5* pred_h, img_height - 1), 0);
			pred[(j*num + i) * 5 + 4] = pred_cls[i*CLASS_NUM + j];
		}
	}
}

void Detector::apply_nms(vector<vector<float> > &pred_boxes, vector<float> &confidence)
{
	for (int i = 0; i < pred_boxes.size() - 1; i++)
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
				if (IOU > NMS_THRESH)
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

int main()
{
	string model_file = "/data/zou/code/py-faster-rcnn-master/models/VGG16/faster_rcnn_alt_opt/faster_rcnn.pt";
	string weights_file = "/data/zou/code/py-faster-rcnn-master/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel";
	int GPUID = 0;
    Caffe::SetDevice(GPUID);
	Caffe::set_mode(Caffe::GPU);
    //Caffe::set_mode(Caffe::CPU);
	Detector det = Detector(model_file, weights_file);
    //string im_names="/data/zou/code/py-faster-rcnn-master/data/demo/004545.jpg";
    clock_t t1 = clock();
    //det.Detect(im_names);
	det.Detect_video("/data/zou/code/video01.avi");
    std::cout << " Time Using GPU-CUDNN: " << (clock() - t1)*1.0 / CLOCKS_PER_SEC << std::endl;
    return 0;
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include "opencv2/imgproc.hpp"

/*
namespace cv{
	static bool sumTemplate(InputArray _src, UMat & result)
	{
		int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
		int wdepth = CV_32F, wtype = CV_MAKE_TYPE(wdepth, cn);
		size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();

		int wgs2_aligned = 1;
		while (wgs2_aligned < (int)wgs)
		    wgs2_aligned <<= 1;
		wgs2_aligned >>= 1;

		char cvt[40];
		ocl::Kernel k("calcSum", ocl::imgproc::match_template_oclsrc,
		              format("-D CALC_SUM -D T=%s -D T1=%s -D WT=%s -D cn=%d -D convertToWT=%s -D WGS=%d -D WGS2_ALIGNED=%d",
		                     ocl::typeToStr(type), ocl::typeToStr(depth), ocl::typeToStr(wtype), cn,
		                     ocl::convertTypeStr(depth, wdepth, cn, cvt),
		                     (int)wgs, wgs2_aligned));
		if (k.empty())
		    return false;

		UMat src = _src.getUMat();
		result.create(1, 1, CV_32FC1);

		ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
		        resarg = ocl::KernelArg::PtrWriteOnly(result);

		k.args(srcarg, src.cols, (int)src.total(), resarg);

		size_t globalsize = wgs;
		return k.run(1, &globalsize, &wgs, false);
	}
	static bool matchTemplateNaive_SQDIFF(InputArray _image, InputArray _templ, OutputArray _result)
	{
		int type = _image.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
		int wdepth = CV_32F, wtype = CV_MAKE_TYPE(wdepth, cn);

		char cvt[40];
		ocl::Kernel k("matchTemplate_Naive_SQDIFF", ocl::imgproc::match_template_oclsrc,
		              format("-D SQDIFF -D T=%s -D T1=%s -D WT=%s -D convertToWT=%s -D cn=%d", ocl::typeToStr(type), ocl::typeToStr(depth),
		              ocl::typeToStr(wtype), ocl::convertTypeStr(depth, wdepth, cn, cvt), cn));
		if (k.empty())
		    return false;

		UMat image = _image.getUMat(), templ = _templ.getUMat();
		_result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
		UMat result = _result.getUMat();

		k.args(ocl::KernelArg::ReadOnlyNoSize(image), ocl::KernelArg::ReadOnly(templ),
		       ocl::KernelArg::WriteOnly(result));

		size_t globalsize[2] = { (size_t)result.cols, (size_t)result.rows };
		return k.run(2, globalsize, NULL, false);
	}

	static bool matchTemplate_SQDIFF(InputArray _image, InputArray _templ, OutputArray _result)
	{
	   
		    matchTemplate(_image, _templ, _result, CV_TM_CCORR);

		    int type = _image.type(), cn = CV_MAT_CN(type);

		    ocl::Kernel k("matchTemplate_Prepared_SQDIFF", ocl::imgproc::match_template_oclsrc,
		              format("-D SQDIFF_PREPARED -D T=%s -D cn=%d", ocl::typeToStr(type),  cn));
		    if (k.empty())
		        return false;

		    UMat image = _image.getUMat(), templ = _templ.getUMat();
		    _result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32F);
		    UMat result = _result.getUMat();

		    UMat image_sums, image_sqsums;
		    integral(image.reshape(1), image_sums, image_sqsums, CV_32F, CV_32F);

		    UMat templ_sqsum;
		    if (!sumTemplate(_templ, templ_sqsum))
		        return false;

		    k.args(ocl::KernelArg::ReadOnlyNoSize(image_sqsums), ocl::KernelArg::ReadWrite(result),
		       templ.rows, templ.cols, ocl::KernelArg::PtrReadOnly(templ_sqsum));

		    size_t globalsize[2] = { (size_t)result.cols, (size_t)result.rows };

		    return k.run(2, globalsize, NULL, false);
		
	}
}
*/
using namespace std;

int main(int argc, char* argv[])
{
	cv::setNumThreads(4);
	cv::ocl::Context context;
	context.create(cv::ocl::Device::TYPE_GPU);
	cv::ocl::Device(context.device(0));
	for(int i = 0; i < context.ndevices(); i++){
		cout << "Device "<<i+1<< " is: " << context.device(i).name() << endl;
	}

//----------------------------------------------
    // calculating time with OpenCL
    
    clock_t t1, t2, t0;
    cv::Mat i,image,image1, recorte1, target;
    i = cv::imread("./real_test3.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	image = i(cv::Rect(0,0,i.cols,i.rows));
	image.convertTo(image,CV_32FC1);
    cv::UMat u = image.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
	cv::UMat v,r;
	image1 = cv::imread("./real_test6.png", CV_LOAD_IMAGE_GRAYSCALE);
	image1.convertTo(image1,CV_32FC1);
	recorte1 = image1(cv::Rect(0,30,image.cols,image.rows-60));
	v = recorte1.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
	r = cv::UMat(cv::Size(1,61), CV_32F, cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
	cout << "type :" << recorte1.type() << endl;



	cv::ocl::setUseOpenCL(true);
	t0 = clock();
	//cv::Canny(u,v,100,50);
	cv::matchTemplate(u,v,r,CV_TM_SQDIFF);
	
    t1 = clock();
    for (int i = 0; i < 1; i++){
        cv::matchTemplate(u,v,r,CV_TM_SQDIFF);
		//cv::Canny(u,v,100,50);
    }
    t2 = clock();
	cout << "Float:"<<endl;
    // showing time with OpenCL
	double diff = ((double)t1 - (double)t0) / CLOCKS_PER_SEC;
	cout << "Running time with OpenCL(first)SQDIFF: \t\t" << diff << " seconds" << endl;
	diff = ((double)t2 - (double)t1) / CLOCKS_PER_SEC;
    cout << "Running time with OpenCL(second)SQDIFF: \t" << diff << " seconds" << endl;


t0 = clock();
	//cv::Canny(u,v,100,50);
	cv::matchTemplate(u,v,r,CV_TM_CCORR);
    t1 = clock();
    for (int i = 0; i < 1; i++){
        cv::matchTemplate(u,v,r,CV_TM_CCORR);
		//cv::Canny(u,v,100,50);
    }
    t2 = clock();

    // showing time with OpenCL
	diff = ((double)t1 - (double)t0) / CLOCKS_PER_SEC;
	cout << "Running time with OpenCL(first)CCORR: \t\t" << diff << " seconds" << endl;
	diff = ((double)t2 - (double)t1) / CLOCKS_PER_SEC;
    cout << "Running time with OpenCL(second)CCORR: \t\t" << diff << " seconds" << endl;
	cout << "size of output is: " << r.size() << endl;

	/*target = r.getMat(cv::ACCESS_READ);
	float* data = (float*)target.data;
	for(int i = 0; i < target.rows; i++){
		cout << i << " : " << *data++ << endl;
	}*/
    /////////////////////////////////////////////////////	
/*
	image.convertTo(image,CV_8UC1);
	image1.convertTo(image1,CV_8UC1);
	u = image.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
	recorte1 = image1(cv::Rect(0,30,image.cols,image.rows-60));
	v = recorte1.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
*/
    // calculating time with OpenCL
    cv::ocl::setUseOpenCL(false);
	t0 = clock();
	//cv::Canny(u,v,100,50);
	cv::matchTemplate(u,v,r,CV_TM_SQDIFF);
    t1 = clock();
    for (int i = 0; i < 1; i++){
        cv::matchTemplate(u,v,r,CV_TM_SQDIFF);
		//cv::Canny(u,v,100,50);
    }
    t2 = clock();

    // showing time with OpenCL
    diff = ((double)t1 - (double)t0) / CLOCKS_PER_SEC;
	cout << "Running time without OpenCL(first)SQDIFF: \t" << diff << " seconds" << endl;
	diff = ((double)t2 - (double)t1) / CLOCKS_PER_SEC;
    cout << "Running time without OpenCL(second)SQDIFF: \t" << diff << " seconds" << endl;

t0 = clock();
	//cv::Canny(u,v,100,50);
	cv::matchTemplate(u,v,r,CV_TM_CCORR);
    t1 = clock();
    for (int i = 0; i < 1; i++){
        cv::matchTemplate(u,v,r,CV_TM_CCORR);
		//cv::Canny(u,v,100,50);
    }
    t2 = clock();

    // showing time with OpenCL
	diff = ((double)t1 - (double)t0) / CLOCKS_PER_SEC;
	cout << "Running time without OpenCL(first)CCORR: \t" << diff << " seconds" << endl;
	diff = ((double)t2 - (double)t1) / CLOCKS_PER_SEC;
    cout << "Running time without OpenCL(second)CCORR: \t" << diff << " seconds" << endl << endl << endl;
	cout << "size of output is: " << r.size() << endl;



	image.convertTo(image,CV_8UC1);
	image1.convertTo(image1,CV_8UC1);
	u = image.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
	recorte1 = image1(cv::Rect(0,30,image.cols,image.rows-60));
	v = recorte1.getUMat(cv::ACCESS_READ,cv::USAGE_ALLOCATE_DEVICE_MEMORY);

	cout<<"UCHAR:\n";

	cv::ocl::setUseOpenCL(true);
	t0 = clock();
	//cv::Canny(u,v,100,50);
	cv::matchTemplate(u,v,r,CV_TM_SQDIFF);
    t1 = clock();
    for (int i = 0; i < 1; i++){
        cv::matchTemplate(u,v,r,CV_TM_SQDIFF);
		//cv::Canny(u,v,100,50);
    }
    t2 = clock();

    // showing time with OpenCL
	diff = ((double)t1 - (double)t0) / CLOCKS_PER_SEC;
	cout << "Running time with OpenCL(first)SQDIFF: \t\t" << diff << " seconds" << endl;
	diff = ((double)t2 - (double)t1) / CLOCKS_PER_SEC;
    cout << "Running time with OpenCL(second)SQDIFF: \t" << diff << " seconds" << endl;


t0 = clock();
	//cv::Canny(u,v,100,50);
	cv::matchTemplate(u,v,r,CV_TM_CCORR);
    t1 = clock();
    for (int i = 0; i < 1; i++){
        cv::matchTemplate(u,v,r,CV_TM_CCORR);
		//cv::Canny(u,v,100,50);
    }
    t2 = clock();

    // showing time with OpenCL
	diff = ((double)t1 - (double)t0) / CLOCKS_PER_SEC;
	cout << "Running time with OpenCL(first)CCORR: \t\t" << diff << " seconds" << endl;
	diff = ((double)t2 - (double)t1) / CLOCKS_PER_SEC;
    cout << "Running time with OpenCL(second)CCORR: \t\t" << diff << " seconds" << endl;
	cout << "size of output is: " << r.size() << endl;

	/*target = r.getMat(cv::ACCESS_READ);
	float* data = (float*)target.data;
	for(int i = 0; i < target.rows; i++){
		cout << i << " : " << *data++ << endl;
	}*/
    /////////////////////////////////////////////////////	

    // calculating time with OpenCL
    cv::ocl::setUseOpenCL(false);
	t0 = clock();
	//cv::Canny(u,v,100,50);
	cv::matchTemplate(u,v,r,CV_TM_SQDIFF);
    t1 = clock();
    for (int i = 0; i < 1; i++){
        cv::matchTemplate(u,v,r,CV_TM_SQDIFF);
		//cv::Canny(u,v,100,50);
    }
    t2 = clock();

    // showing time with OpenCL
    diff = ((double)t1 - (double)t0) / CLOCKS_PER_SEC;
	cout << "Running time without OpenCL(first)SQDIFF: \t" << diff << " seconds" << endl;
	diff = ((double)t2 - (double)t1) / CLOCKS_PER_SEC;
    cout << "Running time without OpenCL(second)SQDIFF: \t" << diff << " seconds" << endl;

t0 = clock();
	//cv::Canny(u,v,100,50);
	cv::matchTemplate(u,v,r,CV_TM_CCORR);
    t1 = clock();
    for (int i = 0; i < 1; i++){
        cv::matchTemplate(u,v,r,CV_TM_CCORR);
		//cv::Canny(u,v,100,50);
    }
    t2 = clock();

    // showing time with OpenCL
	diff = ((double)t1 - (double)t0) / CLOCKS_PER_SEC;
	cout << "Running time without OpenCL(first)CCORR: \t" << diff << " seconds" << endl;
	diff = ((double)t2 - (double)t1) / CLOCKS_PER_SEC;
    cout << "Running time without OpenCL(second)CCORR: \t" << diff << " seconds" << endl << endl << endl;
	cout << "size of output is: " << r.size() << endl;

	/*target = r.getMat(cv::ACCESS_READ);
	data = (float*)target.data;
	for(int i = 0; i < target.rows; i++){
		cout << i << " : " << *data++ << endl;
	}*/
	
	//INFO ABOUT OpenCV BUILD
	//cout << cv::getBuildInformation();
	// INFO ABOUT OpenCL
	cout << "OpenCL: " << endl;
	std::vector<cv::ocl::PlatformInfo> platform_info;
	cv::ocl::getPlatfomsInfo(platform_info);
	for (size_t i = 0; i < platform_info.size(); i++)
	{
		cout
		    << "\tName: " << platform_info[i].name() << endl
		    << "\tVendor: " << platform_info[i].vendor() << endl
		    << "\tVersion: " << platform_info[i].version() << endl
		    << "\tDevice Number: " << platform_info[i].deviceNumber() << endl
		    << endl;
	}

    return 0;
}

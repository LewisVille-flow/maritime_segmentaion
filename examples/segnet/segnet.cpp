/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "jetson-utils/videoSource.h"
#include "jetson-utils/videoOutput.h"

#include "jetson-utils/cudaOverlay.h"
#include "jetson-utils/cudaMappedMemory.h"

#include "jetson-inference/segNet.h"



#include <signal.h>
#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <typeinfo> 


using namespace cv;


#ifdef HEADLESS
	#define IS_HEADLESS() "headless"             // run without display
	#define DEFAULT_VISUALIZATION "overlay"      // output overlay only
#else
	#define IS_HEADLESS() (const char*)NULL      // use display (if attached)
	#define DEFAULT_VISUALIZATION "overlay|mask" // output overlay + mask
#endif


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: segnet [--help] [--network NETWORK] ...\n");
	printf("              input_URI [output_URI]\n\n");
	printf("Segment and classify a video/image stream using a semantic segmentation DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s\n", segNet::Usage());
	printf("%s\n", videoSource::Usage());
	printf("%s\n", videoOutput::Usage());
	printf("%s\n", Log::Usage());

	return 0;
}


//
// segmentation buffers
//
typedef uchar3 pixelType;		// this can be uchar3, uchar4, float3, float4
					// unsigned char를 쓰는 경우 : 적어도 0~255 까지 표현할 수 있습니다
pixelType* imgMask      = NULL;	// color of each segmentation class
pixelType* imgOverlay   = NULL;	// input + alpha-blended mask
pixelType* imgComposite = NULL;	// overlay with mask next to it
pixelType* imgOutput    = NULL;	// reference to one of the above three
pixelType* imgtest	= NULL;

int2 maskSize;
int2 overlaySize;
int2 compositeSize;
int2 outputSize;

// allocate mask/overlay output buffers
bool allocBuffers( int width, int height, uint32_t flags )
{
	// check if the buffers were already allocated for this size
	if( imgOverlay != NULL && width == overlaySize.x && height == overlaySize.y )
		return true;

	// free previous buffers if they exit
	CUDA_FREE_HOST(imgMask);
	CUDA_FREE_HOST(imgOverlay);
	CUDA_FREE_HOST(imgComposite);

	// allocate overlay image
	overlaySize = make_int2(width, height);
	
	if( flags & segNet::VISUALIZE_OVERLAY )
	{
		if( !cudaAllocMapped(&imgOverlay, overlaySize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for overlay image (%ux%u)\n", width, height);
			return false;
		}

		imgOutput = imgOverlay;
		outputSize = overlaySize;
	}

	// allocate mask image (half the size, unless it's the only output)
	if( flags & segNet::VISUALIZE_MASK )
	{
		maskSize = (flags & segNet::VISUALIZE_OVERLAY) ? make_int2(width/2, height/2) : overlaySize;

		if( !cudaAllocMapped(&imgMask, maskSize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for mask image\n");
			return false;
		}

		imgOutput = imgMask;
		outputSize = maskSize;
	}

	// allocate composite image if both overlay and mask are used
	if( (flags & segNet::VISUALIZE_OVERLAY) && (flags & segNet::VISUALIZE_MASK) )
	{
		compositeSize = make_int2(overlaySize.x + maskSize.x, overlaySize.y);

		if( !cudaAllocMapped(&imgComposite, compositeSize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for composite image\n");
			return false;
		}

		imgOutput = imgComposite;
		outputSize = compositeSize;
	}

	return true;
}


int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv, IS_HEADLESS());

	if( cmdLine.GetFlag("help") )
		return usage();


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("segnet:  failed to create input stream\n");
		return 0;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
		LogError("segnet:  failed to create output stream\n");	
	

	/*
	 * create segmentation network
	 */
	segNet* net = segNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("segnet:  failed to initialize segNet\n");
		return 0;
	}

	// set alpha blending value for classes that don't explicitly already have an alpha	
	net->SetOverlayAlpha(cmdLine.GetFloat("alpha", 150.0f));

	// get the desired overlay/mask filtering mode
	const segNet::FilterMode filterMode = segNet::FilterModeFromStr(cmdLine.GetString("filter-mode", "linear"));

	// get the visualization flags
	const uint32_t visualizationFlags = segNet::VisualizationFlagsFromStr(cmdLine.GetString("visualize", DEFAULT_VISUALIZATION));

	// get the object class to ignore (if any)
	const char* ignoreClass = cmdLine.GetString("ignore-class", "void");

	
	printf("before while\n");
	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture next image image
		pixelType* imgInput = NULL;

		if( !input->Capture(&imgInput, 1000) )
		{
			// check for EOS
			if( !input->IsStreaming() )
				break; 

			LogError("segnet:  failed to capture video frame\n");
			continue;
		}
		printf("before allocBuffers\n");
		// allocate buffers for this size frame
		if( !allocBuffers(input->GetWidth(), input->GetHeight(), visualizationFlags) )
		{
			LogError("segnet:  failed to allocate buffers\n");
			continue;
		}
		

		// function test
		std::cout << ">>>>> function test start..." << std::endl << std::endl;
		
		// test for segNet::FindClass()
		std::cout << ">>> segNet::FindClass() test ..." << std::endl;
		int class_tmp = net->FindClassID("car");
		printf("%d \n", class_tmp); // return 15
		
		std::cout << ">>> segNet::GetClassColor() test ..." << std::endl;
		float* class_color = net->GetClassColor(class_tmp);
		printf("\t (%1f, %1f, %1f) \n", class_color[0], class_color[1], class_color[2]); // %d would loses info
		
		// returned (0., 0., 142.) and class 15 has the same rgb value!

		std::cout << ">>>>> function test ends..." << std::endl << std::endl;
		// test ends


		net->Process(imgInput, input->GetWidth(), input->GetHeight(), ignoreClass);
		
		// this is for color mask
		// net->Mask(imgInput, overlaySize.x, overlaySize.y, filterMode);
		
		// added 0509 for binary mask
		net->Mask(imgInput, overlaySize.x, overlaySize.y);
		//
		//std::cout << "imgInput: " << imgInput << std::endl;				// return 0x100fe0000 
		/*
		for( uint32_t y=0; y < overlaySize.y; y++ )
		{
			for( uint32_t x=0; x < overlaySize.x; x++ )
			{
				uchar v = imgInput->x[y * overlaySize.x + x];
				printf(" %d", imgInput->x); // printed 70, imgInput->x type is unsigned char, cannot indexing
			}
		}
		*/
			
		printf(" %d ", imgInput->x);
		printf(" %d ", imgInput->y);
		printf(" %d ", imgInput->z);
		//NOTE: uint_8 = unsigned char(0~255), 8bit 1byte
		//	uchar = unsigned char
		//	imgInput is 'P6uchar3' type


		CUDA(cudaDeviceSynchronize());
		//cv::Mat cv_image(cv::Size(overlaySize.x, overlaySize.y), CV_8UC3, imgInput);
		cv::Mat cv_image(cv::Size(overlaySize.x, overlaySize.y), CV_8U, imgInput);	// CV_8U : 8-bit unsigned integer: uchar
		
		//cv::Mat cv_image_color(cv::Size(overlaySize.x, overlaySize.y), CV_8UC3, imgInput);
		//cv::cvtColor(cv_image_color, cv_image_color, cv::COLOR_RGB2BGR);

		// imgInput is always P6uchar3 type
		//std::cout << "imgInput type: " << typeid(imgInput).name() << std::endl;	// P6uchar3 type
		
		std::cout << "imgInput type: " << typeid(imgInput).name() << std::endl;
		std::cout << "cv_image type: " << typeid(cv_image).name() << std::endl;		// N2cv3MatE type
		
		//printf(imgInput);
		imshow("img", cv_image);
/*
		for (int col = 0; col < overlaySize.y; col++) 
		{ 
			for (int row = 0; row < overlaySize.x; row++) 
		    	{ 
			    	uchar3 v = imgInput[col*16 + row]; 
				printf(" %d", v); 
		    	} 
		    	std::cout << "\n" << std::endl; 
		}
*/
		while(char(waitKey(0)) != 'q');
		

		/* (pixel rgb value test -> failed. 해당 pixel의 값이 0~255라서.)
		unsigned char classIdx_test = cv_image.at<uchar>(282, 378);
		std::cout << "example, cv_image[282][378] is: " << classIdx_test << std::endl; // value F
		printf("example, cv_image[282][378] is: %d \n", classIdx_test);			// value 180


		float* test_color = net->GetClassColor(classIdx_test);		
		printf("test_color (%lf, %lf, %lf)", test_color[0], test_color[1], test_color[2]); // rgb. returned 0 cause classIdx_test 180 is not registered
		*/ 
/*
		for (int row = 0; row < cv_image.rows; row++) 
		{ 
			for (int col = 0; col < cv_image.cols; col++) 
		    	{ 
			    	int v = (int)cv_image.at<uchar>(row, col); 
				printf(" %d", v); 
		    	} 
		    	std::cout << "\n" << std::endl; 
		}
*/

		// std::cout << "row size: " << cv_image.rows << " and col size: " << cv_image.cols << std::endl;
		/*
		uchar b = cv_image_color.at<Vec3b>(282, 380)[0]; //(282, 378) -> (92,91,102) why????
		uchar g = cv_image_color.at<Vec3b>(282, 380)[1]; 
		uchar r = cv_image_color.at<Vec3b>(282, 380)[2]; //(282,380) -> rgb (95, 92, 103) why??
								 // gradient the closer ...
		std::cout << "example, cv_image_color[282][380] color is: " << std::endl; 
		printf("\t (%d, %d, %d)", r, g, b); 
		*/


		
		
		//cv::cvtColor(cv_image, cv_image, cv::COLOR_RGB2BGR);		
		
		/* test 1: test for rgb value of cudasync -> cv Mat image -> cvtcolor -> show 's rgb
		//imshow("img", cv_image);
		for (int row = 0; row < cv_image.rows; row++) 
		{ 
		 	for (int col = 0; col < cv_image.cols; col++) 
		    { 
		    	uchar b = cv_image.at<Vec3b>(row, col)[0]; 
			uchar g = cv_image.at<Vec3b>(row, col)[1]; 
			uchar r = cv_image.at<Vec3b>(row, col)[2]; 
			printf("\t (%d, %d, %d)", r, g, b); 
		    } 
		    std::cout << "\n" << std::endl; 
		}
		// std::cout << "row size: " << cv_image.rows << " and col size: " << cv_image.cols << std::endl;
		// returned: "row size: 565 and col size: 754"
		// rgb examples: (128, 64, 128) -> class 3, label 'road'
		//while(char(waitKey(0)) != 'q');
		*/

		
		/*

		1. FindClassID()와, pixel rgb를 통한 class number 를 통해 class label을 구해보자.
		2. class label을 구했으면 그 클래스의 픽셀 영역을 구해보자
		3. 그 픽셀영역의 대표 값들의 depth를 구해보자
		*/
////////////////////////////////////

		/*
segNet can output two types of mask images - a classID mask, where each pixel is the classID. And the 'colorized' mask where each pixel is the color of the class. So you could set the class colors to grayscale colors. Or you could just use the classID mask and create your own treatment.
Also if you want the class ID's instead of colors, you may want to look at the uint8 version of the segNet::Mask() function, which outputs class ID's:



*/


////////////////////////////////////

		




		//printf(imgInput);		
		//output->Render(imgInput, overlaySize.x, overlaySize.y);

		/*
		printf("before Process\n");
		// process the segmentation network
		if( !net->Process(imgInput, input->GetWidth(), input->GetHeight(), ignoreClass) )
		{
			LogError("segnet:  failed to process segmentation\n");
			continue;
		}
		
		printf("before Overlay\n");
		// generate overlay
		if( visualizationFlags & segNet::VISUALIZE_OVERLAY )
		{
			if( !net->Overlay(imgOverlay, overlaySize.x, overlaySize.y, filterMode) )
			{
				LogError("segnet:  failed to process segmentation overlay.\n");
				continue;
			}
		}

		// generate mask
		if( visualizationFlags & segNet::VISUALIZE_MASK )
		{
			if( !net->Mask(imgMask, maskSize.x, maskSize.y, filterMode) )
			{
				LogError("segnet:-console:  failed to process segmentation mask.\n");
				continue;
			}
		}

		// generate composite
		if( (visualizationFlags & segNet::VISUALIZE_OVERLAY) && (visualizationFlags & segNet::VISUALIZE_MASK) )
		{
			CUDA(cudaOverlay(imgOverlay, overlaySize, imgComposite, compositeSize, 0, 0));
			CUDA(cudaOverlay(imgMask, maskSize, imgComposite, compositeSize, overlaySize.x, 0));
		}

		// render outputs
		if( output != NULL )
		{
			output->Render(imgOutput, outputSize.x, outputSize.y);

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), net->GetNetworkFPS());
			output->SetStatus(str);

			// check if the user quit
			if( !output->IsStreaming() )
				signal_recieved = true;
		}
		*/
		// wait for the GPU to finish		
		CUDA(cudaDeviceSynchronize());
		
		printf("before print out timing info\n");
		// print out timing info
		net->PrintProfilerTimes();
		
	}
	

	/*
	 * destroy resources
	 */
	LogVerbose("segnet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);

	CUDA_FREE_HOST(imgMask);
	CUDA_FREE_HOST(imgOverlay);
	CUDA_FREE_HOST(imgComposite);

	LogVerbose("segnet:  shutdown complete.\n");
	return 0;
}


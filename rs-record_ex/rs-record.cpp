#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>  // Video write

#define OUTPUT_VIDEO_NAME "./0326test_pet2.avi"
#define VIDEO_WINDOW_NAME "video"

using namespace cv;
using namespace std;

int main(int argc, char** argv) try
{
    const auto window_name = "Display Image";
    int flag = 0;
    namedWindow(window_name, WINDOW_AUTOSIZE);

    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : "  << CV_MAJOR_VERSION << endl;
    
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    rs2::config cfg;
    
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    // Start streaming with default recommended configuration
    pipe.start(cfg);

    rs2::align align_to_color(RS2_STREAM_COLOR);
  
    cv::VideoWriter videoWriter;
    

    while (waitKey(10) != 27 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        //cout << "entered" << endl;
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        data = align_to_color.process(data);
        rs2::frame depth_o = data.get_depth_frame().apply_filter(color_map); // applyfilter 이걸 하나 안하나 상관없음

        auto depth = data.get_depth_frame();
        auto color = data.get_color_frame();
        auto colorized_depth = color_map.colorize(depth);
        
        const int w0 = color.as<rs2::video_frame>().get_width();
        const int h0 = color.as<rs2::video_frame>().get_height();
        const int w1 = colorized_depth.as<rs2::video_frame>().get_width();
        const int h1 = colorized_depth.as<rs2::video_frame>().get_height();
	if(flag == 0){
            videoWriter.open(OUTPUT_VIDEO_NAME, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
	    10 , cv::Size(w0+w1, h0), true);
	
	    //영상 저장 셋팅 실패시
	    if (!videoWriter.isOpened())
	    {
		std::cout << "Can't write video !!! check setting" << std::endl;
		return -1;
	    }
	    flag = 1;
	}
		
        Mat image0(Size(w0, h0), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
        //imshow("Display Image", image0);
        Mat image1(Size(w1, h1), CV_8UC3, (void*)colorized_depth.get_data(), Mat::AUTO_STEP);
        //Mat image1(Size(w1, h1), CV_16UC1, (void*)depth.get_data(), Mat::AUTO_STEP);

        Mat conc;
	Mat image;
        //hconcat(image0, image1, conc);
	//imshow(window_name, conc);

        // Query frame size (width and height)
        //const int w = depth_o.as<rs2::video_frame>().get_width();
        //const int h = depth_o.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        //Mat image(Size(w, h), CV_8UC3, (void*)depth_o.get_data(), Mat::AUTO_STEP);
	
        // Update the window with new data
        hconcat(image0, image1, image);
	imshow(window_name, image);

	//받아온 Frame을 저장한다.
	videoWriter << image;

	//'ESC'키를 누르면 종료된다.
	//FPS를 이용하여 영상 재생 속도를 조절하여준다.

	if (cv::waitKey(1000/30) == 27) {
		std::cout << "Stop video record" << std::endl;
		break;
	}


    }

    return EXIT_SUCCESS;
}

catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}



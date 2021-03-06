// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API

using namespace cv;
using namespace std;

int main(int argc, char * argv[]) try
{
    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : "  << CV_MAJOR_VERSION << endl;
    cout << "get~~~ : " << getWindowProperty(window_name, WND_PROP_AUTOSIZE) << endl;
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    
    while (waitKey(10) != 27 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        //cout << "entered" << endl;
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        rs2::frame depth_o = data.get_depth_frame().apply_filter(color_map); // applyfilter 이걸 하나 안하나 상관없음

        auto depth = data.get_depth_frame();
        auto color = data.get_color_frame();
        auto colorized_depth = color_map.colorize(depth);
        
        const int w0 = color.as<rs2::video_frame>().get_width();
        const int h0 = color.as<rs2::video_frame>().get_height();
        const int w1 = colorized_depth.as<rs2::video_frame>().get_width();
        const int h1 = colorized_depth.as<rs2::video_frame>().get_height();
        //cout << "check height1: "  << h1 << endl;

        Mat image0(Size(w0, h0), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
        //imshow("Display Image", image0);
        Mat image1(Size(w1, h1), CV_8UC3, (void*)colorized_depth.get_data(), Mat::AUTO_STEP);
        //Mat image1(Size(w1, h1), CV_16UC1, (void*)depth.get_data(), Mat::AUTO_STEP);

        Mat conc;
	Mat image2;
        //hconcat(image0, image1, conc);
	//imshow(window_name, conc);

        // Query frame size (width and height)
        const int w = depth_o.as<rs2::video_frame>().get_width();
        const int h = depth_o.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        Mat image(Size(w, h), CV_8UC3, (void*)depth_o.get_data(), Mat::AUTO_STEP);

        // Update the window with new data
        hconcat(image0, image1, conc);
        hconcat(conc, image, image2);
	imshow(window_name, image2);

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




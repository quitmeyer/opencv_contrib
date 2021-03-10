/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

						  License Agreement
			   For Open Source Computer Vision Library
					   (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
	this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation
	and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
	may be used to endorse or promote products derived from this software
	without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

#define KEY_UP 72
#define KEY_DOWN 80
#define KEY_LEFT 75
#define KEY_RIGHT 77

// THIS VERSION IS FOR CHESSBOARDS ONLY- SORRY IM SLOPPY WITH CODE

namespace {
	const char* about =
		"Calibration using a CHESS board\n"
		"  To capture a frame for calibration, press 'c',\n"
		"  If input comes from video, press any key for next frame\n"
		"  To finish capturing, press 'ESC' key and calibration starts.\n";
	const char* keys =
		"{w        |   7    | Number of squares in X direction }"
		"{h        |    6   | Number of squares in Y direction }"
		"{sl       |    5   | Square side length (in meters) }"
		"{v        |       | Input from video file, if ommited, input comes from camera }"

		"{ciA       | 1     | Camera id if input doesnt come from video (-v) }"
		"{ciB       | 0     | Camera id if input doesnt come from video (-v) }"

		"{dp       |       | File of marker detector parameters }"
		"{rs       | true | Apply refind strategy }"
		"{zt0       | false | Assume zero tangential distortion }"
		"{zt1       | false | Assume zero tangential distortion }"

		"{a0        |       | Fix aspect ratio (fx/fy) to this value }"
		"{a1        |       | Fix aspect ratio (fx/fy) to this value }"

		"{pc0       | false | Fix the principal point at the center }"
		"{pc1       | false | Fix the principal point at the center }"

		"{sc       | false | Show detected chessboard corners after calibration }"

		"{loadNumImgs       | 46 | number of images to use in the Calibration from Files - If 0 use camera }"
		"{camAfilename       | camA_im | number of images to use in the Calibration from Files }"
		"{camBfilename       | camB_im | number of images to use in the Calibration from Files }"
		"{fileExtension       | .png | number of images to use in the Calibration from Files }"


		;
}

/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters>& params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}



/**
 */
static bool saveCameraParams(const string& filename, Size imageSize, float aspectRatio, int flags,
	const Mat& cameraMatrix, const Mat& distCoeffs, double totalAvgErr) {
	FileStorage fs(filename, FileStorage::WRITE);
	if (!fs.isOpened())
		return false;

	time_t tt;
	time(&tt);
	struct tm* t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_time" << buf;

	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;

	if (flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

	if (flags != 0) {
		sprintf(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;

	return true;
}

static bool saveCameraParamsStereo(const string& filename, Size imageSize, Size imageSize1, float aspectRatio, float aspectRatio1, int flags, int flags1,
	const Mat& cameraMatrix, const Mat& cameraMatrix1, const Mat& distCoeffs, const Mat& distCoeffs1, double totalAvgErr, double totalAvgErr1,
	Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q, double StereoRMS) {
	FileStorage fs(filename, FileStorage::WRITE);
	if (!fs.isOpened())
		return false;

	time_t tt;
	time(&tt);
	struct tm* t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "camA_intrinsics" << cameraMatrix;

	fs << "camA_distorsion" << distCoeffs;

	fs << "camA_size" << imageSize;

	fs << "camB_intrinsics" << cameraMatrix1;

	fs << "camB_distorsion" << distCoeffs1;

	fs << "camB_size" << imageSize;

	fs << "R" << R;
	fs << "T" << T;
	fs<<"R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
	fs << "stereo_error" << StereoRMS;
	fs << "calibration_time" << buf;


	///
	/*
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;

	if (flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

	if (flags != 0) {
		sprintf(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	*/
	return true;
}


/**
 */
int main(int argc, char* argv[]) {

	/*

	TEST
	*/



	//time_t start, end;
	double realfps = 1;

	CommandLineParser parser(argc, argv, keys);
	parser.about(about);

	if (argc < 7) {
		parser.printMessage();
		return 0;
	}

	int squaresX = parser.get<int>("w");
	int squaresY = parser.get<int>("h");
	float squareLength = parser.get<float>("sl");
	//string outputFolder = parser.get<string>("output");
	string outputFolder = "C:/Users/andre/Desktop/Glowcake Hoss/Calibrations/OutputCharuco";
	bool showChessboardCorners = parser.get<bool>("sc");

	int calibrationFlagsA = 0;
	int calibrationFlagsB = 0;

	float aspectRatio = 1;
	float aspectRatio1 = 1;


	int camIdA = parser.get<int>("ciA");
	int camIdB = parser.get<int>("ciB");
	String video;

	String camAfilename = parser.get<String>("camAfilename", true);
	String camBfilename = parser.get<String>("camBfilename", true);
	String fileExtension = parser.get<String>("fileExtension", true);

	
	int loadNumImgs = parser.get<float>("loadNumImgs");


	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	cout << "Initialize Params and or Load Images" << endl;

	cout << "Intialize Boards" << endl;
	//Set up Chessboard Detection
	Size board_size = Size(squaresX, squaresY);
	cout << "board is " << squaresX << "  by  " << squaresY << endl;

	// collect data from each frame
	vector< Mat > allImgsA;
	vector< Mat > allImgsB;
	Size imgSizeA;
	Size imgSizeB;

	vector< vector< Point2f > >  imagePointsA, imagePointsB;
	vector< vector< vector< Point2f > > > allCorners1;

	vector< vector< Point3f > > objectPointsA, objectPointsB;
	vector< vector< int > > allIds1;


	bool foundAq = false;
	bool foundBq = false;
	vector< Point2f > cornersAq, cornersBq;


	//LOAD IMAGES FROM FILE
	if (loadNumImgs > 0) {
		cout << "LOAD IMAGES FROM Stereo FILES " << loadNumImgs << endl;

		for (int i = 0; i < loadNumImgs; i++) { //Load images and DON"T USE LIVE CAMERA
			//char A_img[100], B_img[100];
			//sprintf(A_img, "%s%s%d.%s", outputFolder, camAfilename, i, fileExtension);
			//sprintf(B_img, "%s%s%d.%s", outputFolder, camBfilename, i, fileExtension);
			Mat imgA, imgB;
			string A_img = outputFolder + "/"  + camAfilename + to_string(i) + fileExtension;
			string B_img = outputFolder + "/"  + camBfilename + to_string(i) + fileExtension;

			imgA = imread(A_img, IMREAD_COLOR);
			imgB = imread(B_img, IMREAD_COLOR);

			cout << "Frame " << i << " loaded camA  "<<A_img<<"  |  ";

		

			allImgsA.push_back(imgA);
			imgSizeA = imgA.size();
			cout << "Frame " << i << " loaded camB   " << B_img << endl;

			allImgsB.push_back(imgB);
			imgSizeB = imgB.size();


			/// Double check our loaded images

			//Shrink the Image for Display purposes
			Size showsize;
			//showsize = Size(960, 540);
			showsize = Size(640, 480);
			Mat imageCopyLowResA, imageCopyLowResB;

			imgA.copyTo(imageCopyLowResA);
			imgB.copyTo(imageCopyLowResB);
			resize(imageCopyLowResA, imageCopyLowResA, showsize, 0, 0);
			resize(imageCopyLowResB, imageCopyLowResB, showsize, 0, 0);


			//Detect Chessboards


			foundAq = cv::findChessboardCorners(imageCopyLowResA, board_size, cornersAq, CALIB_CB_FAST_CHECK);

			
			foundBq = cv::findChessboardCorners(imageCopyLowResB, board_size, cornersBq, CALIB_CB_FAST_CHECK);

			if (!foundAq || !foundBq) //skip if we dont get a chessboard, extra check
			{

				cout << "small size error on " << i << "   no chessboard found.   found a and b   " << foundAq << foundBq << endl;

			}
			else
			{
				cout << "Found Board small size  " << i << endl;
				drawChessboardCorners(imageCopyLowResA, board_size, cornersAq, foundAq);

				imshow("test loaded imgs", imageCopyLowResA);
				waitKey(1);
			}

		}
		//imshow("test", allImgsA[0]);
		cout << "Image Sizes " << imgSizeA << "   images b   " << imgSizeB << endl;

	}

	//Do Live Image Capture

	else {

		VideoCapture inputVideoA;
		VideoCapture inputVideoB;

		int waitTime;
		if (!video.empty()) {
			cout << "Capture from Video Frames " << loadNumImgs << endl;
			inputVideoA.open(video);
			waitTime = 0;
		}
		else {
			cout << "Live Capture Images " << loadNumImgs << endl;

			//inputVideoA.open(camIdA, CAP_DSHOW);
			//inputVideoB.open(camIdB, CAP_DSHOW);
			inputVideoA.open(camIdA);
			inputVideoB.open(camIdB); //  runs a bit faster without DSHOW

			/*inputVideoA.open(camIdA, CAP_FFMPEG);
			inputVideoB.open(camIdB, CAP_ANY);*/

			waitTime = 2;
		}



		//inputVideoA.set(CAP_PROP_FOURCC, VideoWriter::fourcc('H' , '2', '6', '4'));			//Camera Settings Dialog

		inputVideoA.set(CAP_PROP_SETTINGS, 0); //This pops up the nice dialog to keep camera settings persistent. You need directshow DSHOW enabled as the capturer, and you need a number here that doesn't do anything but you have to have it there
		inputVideoB.set(CAP_PROP_SETTINGS, 0);
		inputVideoA.set(CAP_PROP_MONOCHROME, 1);

		//inputVideo0.set(CAP_PROP_AUTO_EXPOSURE, .25);
		//inputVideo1.set(CAP_PROP_AUTO_EXPOSURE, .1);

		//inputVideoA.set(CAP_PROP_FPS, 5);

		//inputVideoB.set(CAP_PROP_FPS, 5);


		//Manually Set Camera Parameters

		/**/
		inputVideoA.set(CAP_PROP_FRAME_WIDTH, 3264);
		inputVideoA.set(CAP_PROP_FRAME_HEIGHT, 2448);
		inputVideoB.set(CAP_PROP_FRAME_WIDTH, 3264);
		inputVideoB.set(CAP_PROP_FRAME_HEIGHT, 2448);
		/**/

		cout << "Cameras Started" << endl;
		cout << "Cameras A Properties " << " ID num " << camIdA << " exposure " << inputVideoA.get(CAP_PROP_EXPOSURE) << "  Backend API " << inputVideoA.get(CAP_PROP_BACKEND) << "  Width and Height " << inputVideoA.get(CAP_PROP_FRAME_WIDTH) << " " << inputVideoA.get(CAP_PROP_FRAME_HEIGHT) << endl;
		cout << "Cameras B Properties " << " ID num " << camIdB << " exposure " << inputVideoB.get(CAP_PROP_EXPOSURE) << "  Width and Height " << inputVideoB.get(CAP_PROP_FRAME_WIDTH) << " " << inputVideoB.get(CAP_PROP_FRAME_HEIGHT) << endl;



		cout << "Create Windows" << endl;

		namedWindow("CamA_StereoCalib_Output", WINDOW_KEEPRATIO);
		moveWindow("CamA_StereoCalib_Output", 0, 10);
		resizeWindow("CamA_StereoCalib_Output", 1920, 540);
		int framenum = 0;

		//This is the main video-grabbing loop
		while (1) //inputVideoA.grab() && inputVideoB.grab()) // grab frams at the same time! for multicam
		{
			//while(1){
				//Start the FPS timer
			int64 tickstart = cv::getTickCount();

			Mat imageA, imageCopyLowResA, grayA;

			if (inputVideoA.isOpened())
			{
				Mat viewA;
				inputVideoA >> viewA;
				viewA.copyTo(imageA);
			}

			//inputVideoA.retrieve(imageA);

			Mat imageB, imageCopyLowResB, grayB;
			if (inputVideoB.isOpened())
			{
				Mat viewB;
				inputVideoB >> viewB;
				viewB.copyTo(imageB);
			}

			//inputVideoB.retrieve(imageB);

			//inputVideoA >> imageA;
			//inputVideoB >> imageB;

			vector< int > ids;
			vector< Point2f > cornersA, cornersB, rejected;

			vector< int > ids1;
			vector< vector< Point2f > > corners1, rejected1;

			//MAKE GRAYSCALE FOR CornerSubPix PERFORMANCE
			//cvtColor(imageA, grayA, COLOR_BGR2GRAY);
			//cvtColor(imageB, grayB, COLOR_BGR2GRAY);

			//First search for corners at LOW RES while live streaming

				//Shrink the Image for Display purposes
			Size showsize;
			//showsize = Size(960, 540);
			showsize = Size(640, 480);

			imageA.copyTo(imageCopyLowResA);
			imageB.copyTo(imageCopyLowResB);
			resize(imageCopyLowResA, imageCopyLowResA, showsize, 0, 0);
			resize(imageCopyLowResB, imageCopyLowResB, showsize, 0, 0);



			//Detect Chessboards

			bool foundA = false;
			foundA = cv::findChessboardCorners(imageCopyLowResA, board_size, cornersA, CALIB_CB_FAST_CHECK);

			bool foundB = false;
			foundB = cv::findChessboardCorners(imageCopyLowResB, board_size, cornersB, CALIB_CB_FAST_CHECK);

			
			//Change the gray back to color
		/*	cvtColor(imageCopyA, imageCopyA, COLOR_GRAY2BGR);
			cvtColor(imageCopyB, imageCopyB, COLOR_GRAY2BGR);*/


			if (foundA)
			{
				//cornerSubPix(grayA, corners, cv::Size(5, 5), cv::Size(-1, -1), 				TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
				drawChessboardCorners(imageCopyLowResA, board_size, cornersA, foundA);
			}
			if (foundB)
			{
				//	cornerSubPix(grayB, corners, cv::Size(5, 5), cv::Size(-1, -1), 				TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
				drawChessboardCorners(imageCopyLowResB, board_size, cornersB, foundB);
			}

			Scalar texColA = Scalar(255, 0, 0);
			Scalar texColB = Scalar(255, 0, 0);

			if (!foundA) {
				texColA = Scalar(0, 0, 255);
				putText(imageCopyLowResA, "NOT ALL POINTS VISIBLE ",
					Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.4, texColA, 4);

				//show red detected dots
				drawChessboardCorners(imageCopyLowResA, board_size, cornersA, foundA);

			}
			putText(imageCopyLowResA, "Cam A: Press 'c' to add current frame. 'ESC' to finish and calibrate",
				Point(10, 20), FONT_HERSHEY_SIMPLEX, .5, texColA, 2);


			if (!foundB) {
				texColB = Scalar(0, 0, 255);

				putText(imageCopyLowResB, "NOT ALL POINTS VISIBLE ",
					Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.4, texColB, 2);

				//show red detected dots
				drawChessboardCorners(imageCopyLowResB, board_size, cornersB, foundB);

			}
			putText(imageCopyLowResB, "Cam B: 'c'=add current frame. 'ESC'= calibrate",
				Point(10, 20), FONT_HERSHEY_SIMPLEX, .5, texColB, 2);


			//Show the FPS
			putText(imageCopyLowResA, "FPS: " + to_string(realfps),
				Point(10, 400), FONT_HERSHEY_SIMPLEX, 1, texColA, 2);
			putText(imageCopyLowResB, "FPS: " + to_string(realfps),
				Point(10, 400), FONT_HERSHEY_SIMPLEX, 1, texColA, 2);

			//TODO scale these windows to be more manageable sizes
			//Put windows next to each other
			hconcat(imageCopyLowResA, imageCopyLowResB, imageCopyLowResA);
			imshow("CamA_StereoCalib_Output", imageCopyLowResA);

			//imshow("CamB_StereoCalib_Output", imageCopyB);

			//HANDLE INPUT 

			char key = (char)waitKey(waitTime);

			if (key == 27) {//Leave this loop if we hit escape
			
				//Kill the Cameras

				inputVideoA.release();
				inputVideoB.release();

				//Save all the Captured Images, keep them in the vault
				cout << "Saving All images" << endl;

				bool save1 = false;
				bool save2 = false;

				for (int i = 0; i < allImgsA.size(); i++) {
					ostringstream name;
					//name << i + 1;
					name << i;
					save1 = imwrite(outputFolder + "/" + "camA_im" + name.str() + ".png", allImgsA[i]);
					save2 = imwrite(outputFolder + "/" + "camB_im" + name.str() + ".png", allImgsB[i]);
					if ((save1) && (save2))
					{
						cout << "pattern camA and camB images number " << i  << " saved" << endl << endl;

					}
					else
					{
						cout << "pattern camA and camB images number " << i << " NOT saved" << endl << endl << "Retry, check the path" << endl << endl;
					}
				}

				break;
			}
			//Change Camera around if we need
			if (key == '0') {
				inputVideoA.open(0);
				cout << "Changed CamA to Input 0 " << endl;

			}
			if (key == '1') {
				inputVideoA.open(1);
				cout << "Changed CamA to Input 1 " << endl;

			}
			if (key == '2') {
				inputVideoA.open(2);
				cout << "Changed CamA to Input 2 " << endl;

			}
			if (key == '3') {
				inputVideoA.open(3);
				cout << "Changed CamA to Input 3 " << endl;

			}

			if (key == KEY_LEFT) {
				inputVideoB.open(0);
				cout << "Changed CamB to Input 0 " << endl;

			}
			if (key == KEY_DOWN) {
				inputVideoB.open(1);
				cout << "Changed CamB to Input 1 " << endl;

			}
			if (key == KEY_RIGHT) {
				inputVideoB.open(2);
				cout << "Changed CamB to Input 2 " << endl;

			}

			if (key == 'c' && (!foundA || !foundB)) {

				cout << "Frame Not Captured, Please make sure all IDs are visible!" << endl;


			}

			if (key == 'c' && foundA && foundB) {


				//Process the Captured Frame Chess corners



				cout << "Frame " << framenum << " captured camA" << endl;

				allImgsA.push_back(imageA);
				imgSizeA = imageA.size();



				//Cam B

				cout << "Frame captured camB" << endl;

				allImgsB.push_back(imageB);
				imgSizeB = imageB.size();

				framenum++;

			}
			//Calculate Framerate

			realfps = cv::getTickFrequency() / (cv::getTickCount() - tickstart);

			//cout << "Estimated frames per second : " << realfps << "   Time taken : " << cv::getTickCount() - tickstart << " seconds" << endl;

		}

	}


	/*
	******
	PROCESS STAGE
	******
	*/

	

	

	vector< Point2f > cornersA, cornersB;

	//imshow("Sample Img", allImgsA[0]);

	//Calibrate those chessboards individually!
	cout << "Calibrating Cam A and Cam B at Full Resolution | total images= "<< allImgsA.size() << endl;
	for (int i = 0; i < allImgsA.size(); i++) {

		//Show images as processing for debugging
		Size showsize;
		//showsize = Size(960, 540);
		showsize = Size(640, 480);
		Mat imageCopyLowResA, imageCopyLowResB;

		//allImgsA[i].copyTo(imageCopyLowResA);
		//allImgsB[i].copyTo(imageCopyLowResB);
		//resize(imageCopyLowResA, imageCopyLowResA, showsize, 0, 0);
		//resize(imageCopyLowResB, imageCopyLowResB, showsize, 0, 0);

		//hconcat(imageCopyLowResA, imageCopyLowResB, imageCopyLowResA);
		//imshow("DebugViewCamA_StereoCalib_Output", imageCopyLowResA);

		bool foundAFull = false;
		//foundAFull = cv::findChessboardCorners(allImgsA[i], board_size, cornersA, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS); //CALIB_CB_ADAPTIVE_THRESH | | CALIB_CB_NORMALIZE_IMAGE
		foundAFull = cv::findChessboardCornersSB(allImgsA[i], board_size, cornersA, CALIB_CB_ACCURACY); //CALIB_CB_ADAPTIVE_THRESH | | CALIB_CB_NORMALIZE_IMAGE

		bool foundBFull = false;
		//foundBFull = cv::findChessboardCorners(allImgsB[i], board_size, cornersB, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);
		foundBFull = cv::findChessboardCornersSB(allImgsB[i], board_size, cornersB, CALIB_CB_ACCURACY);

		if (!foundAFull || !foundBFull) //skip if we dont get a chessboard, extra check
		{

			cout << "error on "<<i<< "   no chessboard found.   found a and b   " << foundAFull << foundBFull << endl;

			//drawChessboardCorners(imageCopyLowResA, board_size, cornersA, foundA);
		}
		else
		{
			cout << "Found Board  " << i << endl;

			Mat grayA, grayB;

			/* no Cornersubpix if using findchessboardcornersSB
			cvtColor(allImgsA[i], grayA, COLOR_BGR2GRAY);
			cvtColor(allImgsB[i], grayB, COLOR_BGR2GRAY);
			cornerSubPix(grayA, cornersA, cv::Size(5, 5), cv::Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
			cornerSubPix(grayB, cornersB, cv::Size(5, 5), cv::Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
			*/
			//Chessboard object
			vector< Point3f > obj;
			for (int i = 0; i < squaresY; i++)
				for (int j = 0; j < squaresX; j++)
					obj.push_back(Point3f((float)j * squareLength, (float)i * squareLength, 0));

			//Save all the points of this frame
			imagePointsA.push_back(cornersA);
			objectPointsA.push_back(obj);

			imagePointsB.push_back(cornersB);
			objectPointsB.push_back(obj);

			cout << " Success processed frame  " << i << endl;

		}

	}

	//Calibrate each camera based on the detected points

	Mat cameraMatrixA, distCoeffsA;
	vector< Mat > rvecsA, tvecsA;
	double repErrorA;

	Mat cameraMatrixB, distCoeffsB;
	vector< Mat > rvecsB, tvecsB;
	double repErrorB;

	if (calibrationFlagsA & CALIB_FIX_ASPECT_RATIO) {
		cameraMatrixA = Mat::eye(3, 3, CV_64F);
		cameraMatrixA.at< double >(0, 0) = aspectRatio;
	}

	if (calibrationFlagsB & CALIB_FIX_ASPECT_RATIO) {
		cameraMatrixB = Mat::eye(3, 3, CV_64F);
		cameraMatrixB.at< double >(0, 0) = aspectRatio1;
	}


	int flag = 0;
	flag |= CALIB_FIX_K4;
	flag |= CALIB_FIX_K5;

	repErrorA = calibrateCamera(objectPointsA, imagePointsA, allImgsA[0].size(), cameraMatrixA, distCoeffsA, rvecsA, tvecsA, flag);

	cout << "Cam Matrix A:  " << cameraMatrixA << "  Calibration error Cam A: " << repErrorA << endl;

	repErrorB = calibrateCamera(objectPointsB, imagePointsB, allImgsB[0].size(), cameraMatrixB, distCoeffsB, rvecsB, tvecsB, flag);

	cout << "Cam Matrix B:  " << cameraMatrixB << "  Calibration error Cam A: " << repErrorB << endl;


	//Save Files
	bool saveOk = saveCameraParams(outputFolder + "/" + "_CamA.yml", imgSizeA, aspectRatio, calibrationFlagsA,
		cameraMatrixA, distCoeffsA, repErrorA);

	bool saveOk1 = saveCameraParams(outputFolder + "/" + "_CamB.yml", imgSizeB, aspectRatio1, calibrationFlagsB,
		cameraMatrixB, distCoeffsB, repErrorB);

	if (!saveOk) {
		cerr << "Cannot save output file CAMA" << endl;
		return 0;
	}

	if (!saveOk1) {
		cerr << "Cannot save output file CAMB" << endl;
		return 0;
	}

	cout << "CamA Rep Error: " << repErrorA << endl;
	cout << "CamA Calibration saved to " << outputFolder + "_CamA.yml" << endl;

	cout << "CamB Rep Error: " << repErrorB << endl;
	cout << "CamB Calibration saved to " << outputFolder + "_CamB.yml" << endl;


	//Perform the Stereo Calibration between the Cameras

	cout << "Starting STEREO CALIBRATION Steps " << endl;

	/* STEREO CALIBRATION

	*/

	Mat R, T, E, F;

	//vector< vector< Point3f > > object_points;
	vector< Point2f > corners1, corners2;
	vector< vector< Point2f > > left_img_points, right_img_points;
	vector< Point3f > obj;




	//Question: does Stereocalibrate also do the individual calibration?
	double rms = stereoCalibrate(objectPointsA, imagePointsA, imagePointsB,
		cameraMatrixA, distCoeffsA,
		cameraMatrixB, distCoeffsB,
		imgSizeA, R, T, E, F);


	//SAVE ALL THE STEREO DATA

	cout << "Stereo Calibration done with RMS error=" << rms << endl;
	cout << "Saving Stereo Calibration Files" << endl;
	// save intrinsic parameters
	FileStorage fs(outputFolder + "/" + "Stereo_intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrixA << "D1" << distCoeffsA <<
			"M2" << cameraMatrixB << "D2" << distCoeffsB;
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";


	fs.open(outputFolder + "/" + "Stereo_extrinsics_preRect.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";



	printf("Starting Stereo Rectification\n");
	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrixA, distCoeffsA,
		cameraMatrixB, distCoeffsB,
		imgSizeA, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imgSizeA, &validRoi[0], &validRoi[1]);

	fs.open(outputFolder + "/" + "Stereo_extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	//This is the main file we want to get out of this program for the Structured Light Decoding
	bool saveOkStereo = saveCameraParamsStereo(outputFolder + "/" + "stereoCalibrationParameters_camAcamB.yml", imgSizeA, imgSizeB, aspectRatio, aspectRatio1, calibrationFlagsA, calibrationFlagsB,
		cameraMatrixA, cameraMatrixB, distCoeffsA, distCoeffsB, repErrorA, repErrorB, R, T,R1,R2,P1,P2,Q, rms);

	printf("Done Stereo Rectification\n");




	waitKey();


	destroyAllWindows();
	return 0;
}

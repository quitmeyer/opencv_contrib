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

namespace {
	const char* about =
		"Calibration using a ChArUco board\n"
		"  To capture a frame for calibration, press 'c',\n"
		"  If input comes from video, press any key for next frame\n"
		"  To finish capturing, press 'ESC' key and calibration starts.\n";
	const char* keys =
		"{w        |       | Number of squares in X direction }"
		"{h        |       | Number of squares in Y direction }"
		"{sl       |       | Square side length (in meters) }"
		"{ml       |       | Marker side length (in meters) }"
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
		"{@outfolder |<none> | Output folder to create files of calibrated camera parameters }"
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

		"{sc       | false | Show detected chessboard corners after calibration }";
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
	 Mat R, Mat T, double StereoRMS) {
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
	double realfps=1;

	CommandLineParser parser(argc, argv, keys);
	parser.about(about);

	if (argc < 7) {
		parser.printMessage();
		return 0;
	}

	int squaresX = parser.get<int>("w");
	int squaresY = parser.get<int>("h");
	float squareLength = parser.get<float>("sl");
	float markerLength = parser.get<float>("ml");
	int dictionaryId = parser.get<int>("d");
	string outputFolder = parser.get<string>(0);

	bool showChessboardCorners = parser.get<bool>("sc");

	int calibrationFlags = 0;
	int calibrationFlags1 = 0;

	float aspectRatio = 1;
	float aspectRatio1 = 1;

	if (parser.has("a0")) {
		calibrationFlags |= CALIB_FIX_ASPECT_RATIO;
		aspectRatio = parser.get<float>("a0");
	}
	if (parser.has("a1")) {
		calibrationFlags1 |= CALIB_FIX_ASPECT_RATIO;
		aspectRatio1 = parser.get<float>("a1");
	}
	if (parser.get<bool>("zt0")) calibrationFlags |= CALIB_ZERO_TANGENT_DIST;
	if (parser.get<bool>("pc0")) calibrationFlags |= CALIB_FIX_PRINCIPAL_POINT;

	if (parser.get<bool>("zt1")) calibrationFlags1 |= CALIB_ZERO_TANGENT_DIST;
	if (parser.get<bool>("pc1")) calibrationFlags1 |= CALIB_FIX_PRINCIPAL_POINT;

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
	if (parser.has("dp")) {
		bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
		if (!readOk) {
			cerr << "Invalid detector parameters file" << endl;
			return 0;
		}
	}

	bool refindStrategy = parser.get<bool>("rs");
	int camIdA = parser.get<int>("ciA");
	int camIdB = parser.get<int>("ciB");
	String video;

	if (parser.has("v")) {
		video = parser.get<String>("v");
	}

	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	cout << "Initialize Params" << endl;


	VideoCapture inputVideoA;
	VideoCapture inputVideoB;

	//inputVideoA = VideoCapture(0);
	//inputVideoB = VideoCapture(0);

	int waitTime;
	if (!video.empty()) {
		inputVideoA.open(video);
		waitTime = 0;
	}
	else {
		//inputVideoA.open(camIdA, CAP_DSHOW);
		//inputVideoB.open(camIdB, CAP_DSHOW);
		inputVideoA.open(camIdA);
		inputVideoB.open(camIdB); //  runs a bit faster without DSHOW

		/*inputVideoA.open(camIdA, CAP_FFMPEG);
		inputVideoB.open(camIdB, CAP_ANY);*/

		waitTime = 2;
	}


	//inputVideoA.set(CAP_PROP_FOURCC, VideoWriter::fourcc('H' , '2', '6', '4'));			//Camera Settings Dialog

	inputVideoA.set(CAP_PROP_SETTINGS,0); //This pops up the nice dialog to keep camera settings persistent. You need directshow DSHOW enabled as the capturer, and you need a number here that doesn't do anything but you have to have it there
	inputVideoB.set(CAP_PROP_SETTINGS, 0);
	inputVideoA.set(CAP_PROP_MONOCHROME,1);

	//inputVideo0.set(CAP_PROP_AUTO_EXPOSURE, .25);
	//inputVideo1.set(CAP_PROP_AUTO_EXPOSURE, .1);

	//inputVideoA.set(CAP_PROP_FPS, 5);

	//inputVideoB.set(CAP_PROP_FPS, 5);


	//Manually Set Camera Parameters
	/**
	inputVideoA.set(CAP_PROP_FRAME_WIDTH, 640);
	inputVideoA.set(CAP_PROP_FRAME_HEIGHT, 480);
	inputVideoB.set(CAP_PROP_FRAME_WIDTH, 640);
	inputVideoB.set(CAP_PROP_FRAME_HEIGHT, 480);
	/**/
	/**/
	inputVideoA.set(CAP_PROP_FRAME_WIDTH, 3264);
	inputVideoA.set(CAP_PROP_FRAME_HEIGHT, 2448);
	inputVideoB.set(CAP_PROP_FRAME_WIDTH, 3264);
	inputVideoB.set(CAP_PROP_FRAME_HEIGHT, 2448);
	/**/

	
	

	cout << "Cameras Started" << endl;
	cout << "Cameras A Properties " << " ID num " << camIdA <<" exposure "<< inputVideoA.get(CAP_PROP_EXPOSURE)<< "  Backend API " << inputVideoA.get(CAP_PROP_BACKEND) << "  Width and Height "<< inputVideoA.get(CAP_PROP_FRAME_WIDTH) << " " << inputVideoA.get(CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "Cameras B Properties " << " ID num " << camIdB << " exposure "  << inputVideoB.get(CAP_PROP_EXPOSURE)  << "  Width and Height " << inputVideoB.get(CAP_PROP_FRAME_WIDTH) << " " << inputVideoB.get(CAP_PROP_FRAME_HEIGHT) << endl;


	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	// create charuco board object
	Ptr<aruco::CharucoBoard> charucoboard0 =
		aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
	Ptr<aruco::Board> board0 = charucoboard0.staticCast<aruco::Board>();



	Ptr<aruco::CharucoBoard> charucoboard1 =
		aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
	Ptr<aruco::Board> board1 = charucoboard1.staticCast<aruco::Board>();


	// collect data from each frame
	vector< vector< vector< Point2f > > > allCorners0;
	vector< vector< vector< Point2f > > > allCorners1;

	vector< vector< int > > allIds0;
	vector< vector< int > > allIds1;

	vector< Mat > allImgs0;
	vector< Mat > allImgs1;

	Size imgSize0;
	Size imgSize1;
	cout << "Create Windows" << endl;

	namedWindow("CamA_StereoCalib_Output", WINDOW_KEEPRATIO);
    moveWindow("CamA_StereoCalib_Output", 0, 10);
	resizeWindow("CamA_StereoCalib_Output", 1920, 540);

	/*namedWindow("CamB_StereoCalib_Output", WINDOW_KEEPRATIO);
    moveWindow("CamB_StereoCalib_Output", 960, 10);
	resizeWindow("CamB_StereoCalib_Output", 960, 540);
	*/

	cout << "Intialize Boards" << endl;

	int totalCorners = charucoboard0->chessboardCorners.size();
	cout << "total corners " << totalCorners << endl;

	int framenum = 0;

	//This is the main video-grabbing loop
	while (inputVideoA.grab() && inputVideoB.grab()) // grab frams at the same time! for multicam
	{
	//while(1){
		//Start the FPS timer
		int64 tickstart = cv::getTickCount();
		
		Mat imageA, imageCopyA;
		inputVideoA.retrieve(imageA);

		Mat imageB, imageCopyB;
		inputVideoB.retrieve(imageB);

		//inputVideoA >> imageA;
		//inputVideoB >> imageB;

		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;

		vector< int > ids1;
		vector< vector< Point2f > > corners1, rejected1;

		//MAKE GRAYSCALE FOR PERFORMANCE
	/*	cvtColor(imageA, imageA, COLOR_BGR2GRAY);
		cvtColor(imageB, imageB, COLOR_BGR2GRAY);*/


		// detect markers
		aruco::detectMarkers(imageA, dictionary, corners, ids, detectorParams, rejected);
		aruco::detectMarkers(imageB, dictionary, corners1, ids1, detectorParams, rejected1);


		// refind strategy to detect more markers
		if (refindStrategy) aruco::refineDetectedMarkers(imageA, board0, corners, ids, rejected);

		if (refindStrategy) aruco::refineDetectedMarkers(imageB, board1, corners1, ids1, rejected1);

		// interpolate charuco corners
		Mat currentCharucoCornersA, currentCharucoIdsA;
		Mat currentCharucoCornersB, currentCharucoIdsB;


		if (ids.size() > 0)
			aruco::interpolateCornersCharuco(corners, ids, imageA, charucoboard0, currentCharucoCornersA,
				currentCharucoIdsA);

		if (ids1.size() > 0)
			aruco::interpolateCornersCharuco(corners1, ids1, imageB, charucoboard1, currentCharucoCornersB,
				currentCharucoIdsB);
		
		imageA.copyTo(imageCopyA);
		imageB.copyTo(imageCopyB);

		//Change the gray back to color
	/*	cvtColor(imageCopyA, imageCopyA, COLOR_GRAY2BGR);
		cvtColor(imageCopyB, imageCopyB, COLOR_GRAY2BGR);*/


		// draw results
		/**/

		if (ids.size() > 0) aruco::drawDetectedMarkers(imageCopyA, corners);

		
		if (ids1.size() > 0) aruco::drawDetectedMarkers(imageCopyB, corners1);

		//Draw Charuco markers

		if (currentCharucoCornersA.total() > 0)
				aruco::drawDetectedCornersCharuco(imageCopyA, currentCharucoCornersA, currentCharucoIdsA);
			if (currentCharucoCornersB.total() > 0)
			aruco::drawDetectedCornersCharuco(imageCopyB, currentCharucoCornersB, currentCharucoIdsB, Scalar(255, 255, 255));
	
			/**/
		//Shrink the Image for Display purposes
		Size showsize;
		showsize=  Size(960,540);

		resize(imageCopyA, imageCopyA, showsize, 0, 0);
		resize(imageCopyB, imageCopyB, showsize, 0, 0);

		Scalar texColA = Scalar(255, 0, 0);
		Scalar texColB = Scalar(255, 0, 0);

		if (currentCharucoCornersA.total() < board0->objPoints.size()){
			texColA = Scalar(0, 0, 255);
			putText(imageCopyA, "NOT ALL POINTS VISIBLE " + to_string(currentCharucoCornersA.total()),
				Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.4, texColA, 4);
		}
		putText(imageCopyA, "Cam A: Press 'c' to add current frame. 'ESC' to finish and calibrate",
			Point(10, 20), FONT_HERSHEY_SIMPLEX, .5, texColA, 2);


		if (currentCharucoCornersB.total() < board1->objPoints.size()) {
			texColB = Scalar(0, 0, 255);

			putText(imageCopyB, "NOT ALL POINTS VISIBLE " + to_string(currentCharucoCornersB.total()),
				Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.4, texColB, 2);
		}
		putText(imageCopyB, "Cam B: 'c'=add current frame. 'ESC'= calibrate",
			Point(10, 20), FONT_HERSHEY_SIMPLEX, .5, texColB, 2);


		//Show the FPS
		putText(imageCopyA, "FPS: " + to_string(realfps),
			Point(10, 400), FONT_HERSHEY_SIMPLEX, 1, texColA, 2);
		putText(imageCopyB, "FPS: " + to_string(realfps),
			Point(10, 400), FONT_HERSHEY_SIMPLEX, 1, texColA, 2);

		//TODO scale these windows to be more manageable sizes

		hconcat(imageCopyA, imageCopyB, imageCopyA);
		imshow("CamA_StereoCalib_Output", imageCopyA);

		//imshow("CamB_StereoCalib_Output", imageCopyB);


		char key = (char)waitKey(waitTime);

		if (key == 27) break;//Leave this loop if we hit escape

		//Change Camera around if we need
		if (key == '0') {
			inputVideoA.open(0);
			cout << "Changed CamA to Input 0 "  << endl;

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

		if (key == 'c' && ( currentCharucoCornersB.total() < totalCorners || currentCharucoCornersA.total() < totalCorners)) {

			cout << "Frame Not Captured, Please make sure all IDs are visible!" << endl;


		}

		if (key == 'c' && currentCharucoCornersB.total() == totalCorners && currentCharucoCornersA.total() == totalCorners) {
			cout << "Frame "<< framenum <<" captured camA" << endl;
			allCorners0.push_back(corners);
			allIds0.push_back(ids);
			allImgs0.push_back(imageA);
			imgSize0 = imageA.size();

			
				cout << "Frame captured camB" << endl;
				allCorners1.push_back(corners1);
				allIds1.push_back(ids1);
				allImgs1.push_back(imageB);
				imgSize1 = imageB.size();
		
			framenum++;

		}
		//Calculate Framerate

		realfps = cv::getTickFrequency() / (cv::getTickCount() - tickstart);

		//cout << "Estimated frames per second : " << realfps << "   Time taken : " << cv::getTickCount() - tickstart << " seconds" << endl;

	}

	cout << "Saving All images" << endl;


	bool save1 = false;
	bool save2 = false;

	for (int i = 0; i < allImgs0.size(); i++) {
		ostringstream name;
		name << i + 1;
		save1 = imwrite(outputFolder + "/" + "camA_im" + name.str() + ".png", allImgs0[i]);
		save2 = imwrite(outputFolder + "/" + "camB_im" + name.str() + ".png", allImgs1[i]);
		if ((save1) && (save2))
		{
			cout << "pattern camA and camB images number " << i + 1 << " saved" << endl << endl;

		}
		else
		{
			cout << "pattern camA and camB images number " << i + 1 << " NOT saved" << endl << endl << "Retry, check the path" << endl << endl;
		}
	}




	cout << "Video Grabbing ended, Starting to Process" << endl;

	if (allIds0.size() < 1) {
		cerr << "Not enough captures for calibration" << endl;
		return 0;
	}
	if (allIds1.size() < 1) {
		cerr << "Not enough captures for calibration" << endl;
		return 0;
	}

	Mat cameraMatrix, distCoeffs;
	vector< Mat > rvecs, tvecs;
	double repError;

	Mat cameraMatrix1, distCoeffs1;
	vector< Mat > rvecs1, tvecs1;
	double repError1;

	if (calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
		cameraMatrix = Mat::eye(3, 3, CV_64F);
		cameraMatrix.at< double >(0, 0) = aspectRatio;
	}

	if (calibrationFlags1 & CALIB_FIX_ASPECT_RATIO) {
		cameraMatrix1 = Mat::eye(3, 3, CV_64F);
		cameraMatrix1.at< double >(0, 0) = aspectRatio1;
	}

	// prepare data for calibration
	vector< vector< Point2f > > allCornersConcatenated;
	vector< int > allIdsConcatenated;
	vector< int > markerCounterPerFrame;

	vector< vector< Point2f > > allCornersConcatenated1;
	vector< int > allIdsConcatenated1;
	vector< int > markerCounterPerFrame1;

	markerCounterPerFrame.reserve(allCorners0.size());
	for (unsigned int i = 0; i < allCorners0.size(); i++) {
		markerCounterPerFrame.push_back((int)allCorners0[i].size());
		for (unsigned int j = 0; j < allCorners0[i].size(); j++) {
			allCornersConcatenated.push_back(allCorners0[i][j]);
			allIdsConcatenated.push_back(allIds0[i][j]);
		}
	}

	markerCounterPerFrame1.reserve(allCorners1.size());
	for (unsigned int i = 0; i < allCorners1.size(); i++) {
		markerCounterPerFrame1.push_back((int)allCorners1[i].size());
		for (unsigned int j = 0; j < allCorners1[i].size(); j++) {
			allCornersConcatenated1.push_back(allCorners1[i][j]);
			allIdsConcatenated1.push_back(allIds1[i][j]);
		}
	}

	// calibrate camera using aruco markers
	double arucoRepErr;
	arucoRepErr = aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
		markerCounterPerFrame, board0, imgSize0, cameraMatrix,
		distCoeffs, noArray(), noArray(), calibrationFlags);

	double arucoRepErr1;
	arucoRepErr1 = aruco::calibrateCameraAruco(allCornersConcatenated1, allIdsConcatenated1,
		markerCounterPerFrame1, board1, imgSize1, cameraMatrix1,
		distCoeffs1, noArray(), noArray(), calibrationFlags1);

	// prepare data for charuco calibration
	int nFrames = (int)allCorners0.size();
	int nFrames1 = (int)allCorners1.size();

	vector< Mat > allCharucoCorners;
	vector< Mat > allCharucoIds;
	vector< Mat > filteredImages;
	allCharucoCorners.reserve(nFrames);
	allCharucoIds.reserve(nFrames);

	vector< Mat > allCharucoCorners1;
	vector< Mat > allCharucoIds1;
	vector< Mat > filteredImages1;
	allCharucoCorners1.reserve(nFrames1);
	allCharucoIds1.reserve(nFrames1);

	for (int i = 0; i < nFrames; i++) {
		// interpolate using camera parameters
		Mat currentCharucoCorners, currentCharucoIds;
		aruco::interpolateCornersCharuco(allCorners0[i], allIds0[i], allImgs0[i], charucoboard0,
			currentCharucoCorners, currentCharucoIds, cameraMatrix,
			distCoeffs);

		allCharucoCorners.push_back(currentCharucoCorners);
		allCharucoIds.push_back(currentCharucoIds);
		filteredImages.push_back(allImgs0[i]);
	}

	for (int i = 0; i < nFrames1; i++) {
		// interpolate using camera parameters
		Mat currentCharucoCorners1, currentCharucoIds1;
		aruco::interpolateCornersCharuco(allCorners1[i], allIds1[i], allImgs1[i], charucoboard1,
			currentCharucoCorners1, currentCharucoIds1, cameraMatrix1,
			distCoeffs);

		allCharucoCorners1.push_back(currentCharucoCorners1);
		allCharucoIds1.push_back(currentCharucoIds1);
		filteredImages1.push_back(allImgs1[i]);
	}

	if (allCharucoCorners.size() < 4) {
		cerr << "Not enough corners on CAMA for calibration" << endl;
		return 0;
	}

	if (allCharucoCorners1.size() < 4) {
		cerr << "Not enough corners on camB for calibration" << endl;
		return 0;
	}

	// calibrate camera using charuco
	repError =
		aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoboard0, imgSize0,
			cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);

	repError1 =
		aruco::calibrateCameraCharuco(allCharucoCorners1, allCharucoIds1, charucoboard1, imgSize1,
			cameraMatrix1, distCoeffs1, rvecs1, tvecs1, calibrationFlags1);

	bool saveOk = saveCameraParams(outputFolder + "/" + "_CamA.yml", imgSize0, aspectRatio, calibrationFlags,
		cameraMatrix, distCoeffs, repError);

	bool saveOk1 = saveCameraParams(outputFolder + "/" + "_CamB.yml", imgSize1, aspectRatio1, calibrationFlags1,
		cameraMatrix1, distCoeffs1, repError1);

	if (!saveOk) {
		cerr << "Cannot save output file CAMA" << endl;
		return 0;
	}

	if (!saveOk1) {
		cerr << "Cannot save output file CAMB" << endl;
		return 0;
	}

	cout << "CamA Rep Error: " << repError << endl;
	cout << "CamA Rep Error Aruco: " << arucoRepErr << endl;
	cout << "CamA Calibration saved to " << outputFolder + "_CamA.yml" << endl;

	cout << "CamB Rep Error: " << repError1 << endl;
	cout << "CamB Rep Error Aruco: " << arucoRepErr1 << endl;
	cout << "CamB Calibration saved to " << outputFolder + "_CamB.yml" << endl;
	/*Perform Stereo Calibration Steps*/

	cout << "Starting STEREO CALIBRATION Steps " << endl;

	/* STEREO CALIBRATION
	   
	*/
	   	 
	Mat R, T, E, F;

	vector< vector< Point3f > > object_points;
	vector< Point2f > corners1, corners2;
	vector< vector< Point2f > > left_img_points, right_img_points;
	vector< Point3f > obj;


	for (int i = 0; i < nFrames; i++)
	{

		object_points.push_back(charucoboard0->chessboardCorners);

	}

	

	double rms = stereoCalibrate(object_points, allCharucoCorners, allCharucoCorners1,
		cameraMatrix, distCoeffs,
		cameraMatrix1, distCoeffs1,
		imgSize0, R, T, E, F);


	//TRY STEREO CALIB ON FIRST FRAME
   /* double rms = stereoCalibrate(objectPoints, imagePoints0, imagePoints0,
		cameraMatrix, distCoeffs,
		cameraMatrix1, distCoeffs1,
		imgSize0, R, T, E, F,
		CALIB_FIX_ASPECT_RATIO +
		CALIB_ZERO_TANGENT_DIST +
		CALIB_USE_INTRINSIC_GUESS +
		CALIB_SAME_FOCAL_LENGTH +
		CALIB_RATIONAL_MODEL +
		CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));*/
	cout << "Stereo Calibration done with RMS error=" << rms << endl;
	cout << "Saving Stereo Calibration Files=" << rms << endl;

	//cv::FileStorage fs1(out_file, cv::FileStorage::WRITE);
	//fs1 << "K1" << K1;
	//fs1 << "K2" << K2;
	//fs1 << "D1" << D1;
	//fs1 << "D2" << D2;
	//fs1 << "R" << R;
	//fs1 << "T" << T;
	//fs1 << "E" << E;
	//fs1 << "F" << F;

	 // save intrinsic parameters
	FileStorage fs(outputFolder + "/" + "Stereo_intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix << "D1" << distCoeffs <<
			"M2" << cameraMatrix1 << "D2" << distCoeffs1;
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";



	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix, distCoeffs,
		cameraMatrix1, distCoeffs1,
		imgSize0, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imgSize0, &validRoi[0], &validRoi[1]);

	fs.open(outputFolder + "/" + "Stereo_extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	//This is the main file we want to get out of this program for the Structured Light Decoding
	bool saveOkStereo = saveCameraParamsStereo(outputFolder + "/" + "stereoCalibrationParameters_camAcamB.yml", imgSize0, imgSize1, aspectRatio, aspectRatio1, calibrationFlags, calibrationFlags1,
		cameraMatrix, cameraMatrix1, distCoeffs, distCoeffs1, repError,repError1, R, T, rms);


	// show interpolated charuco corners for debugging
	if (showChessboardCorners) {
		for (unsigned int frame = 0; frame < filteredImages.size(); frame++) {
			Mat imageCopy = filteredImages[frame].clone();
			if (allIds0[frame].size() > 0) {

				if (allCharucoCorners[frame].total() > 0) {
					aruco::drawDetectedCornersCharuco(imageCopy, allCharucoCorners[frame],
						allCharucoIds[frame]);
				}
			}

			imshow("View Finals_CamA", imageCopy);
			//imshow("View Finals_CamB", imageCopy1);

			char key = (char)waitKey(0);
			if (key == 27) break;
		}
	}

	inputVideoA.release();
	inputVideoB.release();

		destroyAllWindows();
	return 0;
}

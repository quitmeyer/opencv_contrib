/*M///////////////////////////////////////////////////////////////////////////////////////
 //

 //
 //M*/
//#include "DataExporter.cpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/structured_light.hpp>
#include <opencv2/opencv_modules.hpp>
#include <fstream>  

// (if you did not build the opencv_viz module, you will only see the disparity images)
#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#endif
#include <iomanip>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;

//Small projector is 1366 x 768    large projector is 1920 x 1080

static const char* keys =
{ "{@images_list | | Image list where the captured pattern images are saved}"
    "{@calib_param_path     | | Calibration_parameters            }"
    "{@proj_width      | | The projector width used to acquire the pattern          }"
    "{@proj_height     | | The projector height used to acquire the pattern}"
    "{@white_thresh     | | The white threshold height (optional)}"
    "{@black_thresh     | | The black threshold (optional)}" };

static void help()
{
  cout << "\nThis example shows how to use the \"Structured Light module\" to decode a previously acquired gray code pattern, generating a pointcloud"
        "\nCall:\n"
        "./example_structured_light_pointcloud <images_list> <calib_param_path> <proj_width> <proj_height> <white_thresh> <black_thresh>\n"
        << endl;
}

static bool readStringList( const string& filename, vector<string>& l )
{
  l.resize( 0 );
  FileStorage fs( filename, FileStorage::READ );
  if( !fs.isOpened() )
  {
    cerr << "failed to open " << filename << endl;
    return false;
  }
  FileNode n = fs.getFirstTopLevelNode();
  if( n.type() != FileNode::SEQ )
  {
    cerr << "cam 1 images are not a sequence! FAIL" << endl;
    return false;
  }

  FileNodeIterator it = n.begin(), it_end = n.end();
  for( ; it != it_end; ++it )
  {
    l.push_back( ( string ) *it );
  }

  n = fs["cam2"];
  if( n.type() != FileNode::SEQ )
  {
    cerr << "cam 2 images are not a sequence! FAIL" << endl;
    return false;
  }

  it = n.begin(), it_end = n.end();
  for( ; it != it_end; ++it )
  {
    l.push_back( ( string ) *it );
  }

  if( l.size() % 2 != 0 )
  {
    cout << "Error: the image list contains odd (non-even) number of elements\n";
    return false;
  }
  return true;
}


//GUI buttton   This function should be prototyped as void Foo(int state,*void); . state is the current state of the button. It could be -1 for a push button, 0 or 1 for a check/radio box button. 
void callbackButton(int state, void* userdata ) {

}

int main( int argc, char** argv )
{
  structured_light::GrayCodePattern::Params params;
  CommandLineParser parser( argc, argv, keys );
  String images_file = parser.get<String>( 0 );
  String calib_file = parser.get<String>( 1 );

  params.width = parser.get<int>( 2 );
  params.height = parser.get<int>( 3 );

  if( images_file.empty() || calib_file.empty() || params.width < 1 || params.height < 1 || argc < 5 || argc > 7 )
  {
    help();
    return -1;
  }

  
  // Set up GraycodePattern with params
  Ptr<structured_light::GrayCodePattern> graycode = structured_light::GrayCodePattern::create( params );
  size_t white_thresh = 0;
  size_t black_thresh = 0;

  if( argc == 7 )
  {
    // If passed, setting the white and black threshold, otherwise using default values
    white_thresh = parser.get<unsigned>( 4 );
    black_thresh = parser.get<unsigned>( 5 );

    graycode->setWhiteThreshold( white_thresh );
    graycode->setBlackThreshold( black_thresh );
  }

  vector<string> imagelist;
  bool ok = readStringList( images_file, imagelist );
  if( !ok || imagelist.empty() )
  {
    cout << "can not open " << images_file << " or the string list is empty" << endl;
    help();
    return -1;
  }

  FileStorage fs( calib_file, FileStorage::READ );
  if( !fs.isOpened() )
  {
    cout << "Failed to open Calibration Data File." << endl;
    help();
    return -1;
  }

  // Loading calibration parameters
  Mat cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, R, T;
  fs["cam1_intrinsics"] >> cam1intrinsics;
  fs["cam2_intrinsics"] >> cam2intrinsics;
  fs["cam1_distorsion"] >> cam1distCoeffs;
  fs["cam2_distorsion"] >> cam2distCoeffs;
  fs["R"] >> R;
  fs["T"] >> T;

  cout << "cam1intrinsics" << endl << cam1intrinsics << endl;
  cout << "cam1distCoeffs" << endl << cam1distCoeffs << endl;
  cout << "cam2intrinsics" << endl << cam2intrinsics << endl;
  cout << "cam2distCoeffs" << endl << cam2distCoeffs << endl;
  cout << "T" << endl << T << endl << "R" << endl << R << endl;

  if( (!R.data) || (!T.data) || (!cam1intrinsics.data) || (!cam2intrinsics.data) || (!cam1distCoeffs.data) || (!cam2distCoeffs.data) )
  {
    cout << "Failed to load cameras calibration parameters" << endl;
    help();
    return -1;
  }

  size_t numberOfPatternImages = graycode->getNumberOfPatternImages();
  vector<vector<Mat> > captured_pattern;
  captured_pattern.resize( 2 );
  captured_pattern[0].resize( numberOfPatternImages );
  captured_pattern[1].resize( numberOfPatternImages );

  Mat color = imread( imagelist[numberOfPatternImages], IMREAD_COLOR );
  Size imagesSize = color.size();

  // Stereo rectify
  cout << "Rectifying images..." << endl;
  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];
  stereoRectify( cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, imagesSize, R, T, R1, R2, P1, P2, Q, 0,
                -1, imagesSize, &validRoi[0], &validRoi[1] );

  Mat map1x, map1y, map2x, map2y;
  initUndistortRectifyMap( cam1intrinsics, cam1distCoeffs, R1, P1, imagesSize, CV_32FC1, map1x, map1y );
  initUndistortRectifyMap( cam2intrinsics, cam2distCoeffs, R2, P2, imagesSize, CV_32FC1, map2x, map2y );

  namedWindow("Unrectified", WINDOW_NORMAL);

  resizeWindow("Unrectified", 700, 700);
  moveWindow("Unrectified", 0, 10);
  namedWindow("Rectified", WINDOW_NORMAL);

  resizeWindow("Rectified", 700, 700);
  moveWindow("Rectified", 900, 10);



  // Loading pattern images
  for( size_t i = 0; i < numberOfPatternImages; i++ )
  {
      cout <<i <<" of "<< numberOfPatternImages << endl;

      if ((!imread(imagelist[i], IMREAD_GRAYSCALE).data) || (!imread(imagelist[i + numberOfPatternImages + 2], IMREAD_GRAYSCALE).data))
      {
          cout << "Empty images at index " << i << " " << imagelist[i] << "  or  " << imagelist[i + numberOfPatternImages + 2] << endl;
          help();
          return -1;
      }
    captured_pattern[0][i] = imread( imagelist[i], IMREAD_GRAYSCALE );
    captured_pattern[1][i] = imread( imagelist[i + numberOfPatternImages + 2], IMREAD_GRAYSCALE );

 

    remap( captured_pattern[0][i], captured_pattern[0][i], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
    remap(captured_pattern[1][i], captured_pattern[1][i], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar());


    //Debug and see remapped images
    /**
    imshow("Unrectified", imread(imagelist[i + numberOfPatternImages + 2], IMREAD_GRAYSCALE));


    Mat result8u;
    captured_pattern[1][i].convertTo(result8u, CV_8U);
    imshow("Rectified", captured_pattern[1][i]);
    waitKey(0); // have to wait to see image
    /**/
  }
  cout << "done" << endl;

  vector<Mat> blackImages;
  vector<Mat> whiteImages;

  blackImages.resize( 2 );
  whiteImages.resize( 2 );

  // Loading images (all white + all black) needed for shadows computation
  cvtColor( color, whiteImages[0], COLOR_RGB2GRAY );

  whiteImages[1] = imread( imagelist[2 * numberOfPatternImages + 2], IMREAD_GRAYSCALE );
  blackImages[0] = imread( imagelist[numberOfPatternImages + 1], IMREAD_GRAYSCALE );
  blackImages[1] = imread( imagelist[2 * numberOfPatternImages + 2 + 1], IMREAD_GRAYSCALE );

  remap( color, color, map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );

  remap( whiteImages[0], whiteImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( whiteImages[1], whiteImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );

  remap( blackImages[0], blackImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( blackImages[1], blackImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );


  cout << endl << "Decoding pattern ..." << endl;
  Mat disparityMap;
  bool decoded = graycode->decode( captured_pattern, disparityMap, blackImages, whiteImages,
                                  structured_light::DECODE_3D_UNDERWORLD );
  if( decoded )
  {
    cout << endl << "pattern decoded" << endl;

    // To better visualize the result, apply a colormap to the computed disparity
    double min;
    double max;
    minMaxIdx(disparityMap, &min, &max);
    Mat cm_disp, scaledDisparityMap;
    cout << "disp min " << min << endl << "disp max " << max << endl;
    convertScaleAbs( disparityMap, scaledDisparityMap, 255 / ( max - min ) );
    applyColorMap( scaledDisparityMap, cm_disp, COLORMAP_JET );

    // Show the result
    resize( cm_disp, cm_disp, Size( 640*2, 480*2 ), 0, 0, INTER_LINEAR_EXACT );
    imshow( "cm disparity m", cm_disp );


    // Compute the point cloud
    Mat pointCloud;
    disparityMap.convertTo( disparityMap, CV_32FC1 );
    reprojectImageTo3D( disparityMap, pointCloud, Q, true, -1 );
    

    cout << endl << "  Image Reprojected to 3D  " << endl;



    // Compute a mask to remove background
    Mat dst, thresholded_disp;
    threshold( scaledDisparityMap, thresholded_disp, 0, 255, THRESH_OTSU + THRESH_BINARY );
    resize( thresholded_disp, dst, Size( 640*2, 480*2 ), 0, 0, INTER_LINEAR_EXACT );
    imshow( "threshold disp otsu", dst );

    // Apply the mask to the point cloud
    Mat pointcloud_tresh, color_tresh;
    pointCloud.copyTo(pointcloud_tresh, thresholded_disp);
    color.copyTo(color_tresh, thresholded_disp);

    //Try to save Point cloud
    struct dataType { Point3d point; int red; int green; int blue; };
    typedef dataType SpacePoint;
    vector<SpacePoint> pointCloudSpace;

    cv::FileStorage storage("C:/Users/andre/Desktop/Glowcake Hoss/Calibrations/TestDecode/data/pointcloud.mat", cv::FileStorage::WRITE);
    storage << "img" << pointcloud_tresh;
    storage.release();

    //Trying to Save a PLY
    cout << endl << "  Starting to Save the PLY  " << endl;
    unsigned long numElem;
    if (pointcloud_tresh.channels() == 3) {
        numElem = pointcloud_tresh.rows * pointcloud_tresh.cols;
    }
    else {
        numElem = pointcloud_tresh.rows;
    }

    ofstream outfile("C:/Users/andre/Desktop/Glowcake Hoss/Calibrations/TestDecode/data/pointcloud.ply");
    // MARK: Header writing
    outfile << "ply" << std::endl <<
        "format " << "format ascii" << " 1.0" << std::endl <<
        "comment file created using code by Cedric Menard" << std::endl <<
        "element vertex " << numElem << std::endl <<
        "property float x" << std::endl <<
        "property float y" << std::endl <<
        "property float z" << std::endl <<
        "property uchar red" << std::endl <<
        "property uchar green" << std::endl <<
        "property uchar blue" << std::endl <<
        "end_header" << std::endl;


    // Pointer to data
    const float* pData = pointcloud_tresh.ptr<float>(0);
    const unsigned char* pColor = color_tresh.ptr<unsigned char>(0);
    const unsigned long numIter = 3 * numElem;                            // Number of iteration (3 channels * numElem)
   // const bool hostIsLittleEndian = isLittleEndian();

    float_t bufferXYZ;                                                 // Coordinate buffer for float type
    cout << endl << "  loop through  " << endl;

    for (unsigned long i = 0; i < numIter; i += 3) {                            // Loop through all elements
        for (unsigned int j = 0; j < 3; j++) {                                // Loop through 3 coordinates
            outfile << std::setprecision(9) << pData[i + j] << " ";
        }
        for (int j = 2; j >= 0; j--) {
            // OpenCV uses BGR format, so the order of writing is reverse to comply with the RGB format
            outfile << (unsigned short)pColor[i + j] << (j == 0 ? "" : " ");                     // Loop through RGB
        }
        outfile << std::endl;                                            // End if element line
    }
    /*for (int i = 0; i < pointCloud.; i++)
    {

        Point3d point = pointCloud.at(i).point;
        outfile << point.x << " ";
        outfile << point.y << " ";
        outfile << point.z << " ";
        outfile << "\n";
    }
    outfile.close();*/

    outfile.close();
    cout << endl << "  Saved Point cloud to file  " << endl;
    

#ifdef HAVE_OPENCV_VIZ
    // Apply the mask to the point cloud
    Mat pointcloud_tresh, color_tresh;
    pointcloud.copyTo( pointcloud_tresh, thresholded_disp );
    color.copyTo( color_tresh, thresholded_disp );

    // Show the point cloud on viz
    viz::Viz3d myWindow( "Point cloud with color" );
    myWindow.setBackgroundMeshLab();
    myWindow.showWidget( "coosys", viz::WCoordinateSystem() );
    myWindow.showWidget( "pointcloud", viz::WCloud( pointcloud_tresh, color_tresh ) );
    myWindow.showWidget( "text2d", viz::WText( "Point cloud", Point(20, 20), 20, viz::Color::green() ) );
    myWindow.spin();
#endif // HAVE_OPENCV_VIZ

  }

  waitKey();
  return 0;
}

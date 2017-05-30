/** ****************************************************************************
 *  @file    face_headpose_cnn_test.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2017/05
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

#include <trace.hpp>
#include <FaceAnnotation.hpp>
#include <FaceHeadPoseCnn.hpp>
#include <utils.hpp>

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
int
main
  (
  int argc,
  char **argv
  )
{
  // Read sample annotations
  upm::FaceAnnotation ann;
  ann.filename = "test/image68232.jpg";
  ann.bbox.pos = cv::Rect2f(103.0f, 78.0f, 114.0f, 114.0f);
  ann.headpose = cv::Point3f(-18.509f, -3.105f, -17.571f);
  cv::Mat frame = cv::imread(ann.filename, cv::IMREAD_COLOR);
  if (frame.empty())
    return EXIT_FAILURE;

  // Set face detected position
  std::vector<upm::FaceAnnotation> faces(1);
  faces[0].bbox.pos = ann.bbox.pos;

  /// Load face components
  boost::shared_ptr<upm::FaceComposite> composite(new upm::FaceComposite());
  boost::shared_ptr<upm::FaceHeadPose> fh(new upm::FaceHeadPoseCnn("data/"));
  composite->addComponent(fh);

  /// Parse face component options
  composite->parseOptions(argc, argv);
  composite->load();

  // Process frame
  double ticks = processFrame(frame, composite, faces, ann);
  UPM_PRINT("FPS = " << cv::getTickFrequency()/ticks);

  // Evaluate results
  boost::shared_ptr<std::ostream> output(&std::cout, [](std::ostream*){});
  composite->evaluate(output, faces, ann);

  // Draw results
//  boost::shared_ptr<upm::Viewer> viewer(new upm::Viewer);
//  viewer->init(0, 0, "face_headpose_cnn_test");
//  showResults(viewer, ticks, 0, frame, composite, faces, ann);

  UPM_PRINT("End of face_headpose_cnn_test");
  return EXIT_SUCCESS;
};

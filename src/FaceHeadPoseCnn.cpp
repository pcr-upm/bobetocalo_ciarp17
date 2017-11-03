/** ****************************************************************************
 *  @file    FaceHeadPoseCnn.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2015/06
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <utils.hpp>
#include <FaceHeadPoseCnn.hpp>
#include <boost/progress.hpp>
#include <boost/program_options.hpp>

namespace upm {

const float TRAIN_IMAGES_PERCENTAGE = 0.9f;

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
FaceHeadPoseCnn::FaceHeadPoseCnn(std::string path)
{
  _data_path = path;
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceHeadPoseCnn::parseOptions
  (
  int argc,
  char **argv
  )
{
  // Declare the supported program options
  namespace po = boost::program_options;
  po::options_description desc("FaceHeadPoseCnn options");
  UPM_PRINT(desc);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceHeadPoseCnn::train
  (
  const std::vector<upm::FaceAnnotation> &anns
  )
{
  // Select random annotations for training and validation
  std::ofstream ofs_train(_data_path + "train.txt");
  std::ofstream ofs_valid(_data_path + "valid.txt");
  const unsigned int num_data = static_cast<unsigned int>(anns.size());
  int num_train_data = static_cast<int>(num_data * TRAIN_IMAGES_PERCENTAGE);
  for (int i=0; i < num_train_data; i++)
    ofs_train << anns[i].filename << "," << anns[i].bbox.pos.x << "," << anns[i].bbox.pos.y << "," << anns[i].bbox.pos.width << "," << anns[i].bbox.pos.height << "," << anns[i].headpose.x << "," << anns[i].headpose.y << "," << anns[i].headpose.z << std::endl;
  for (int i=num_train_data; i < num_data; i++)
    ofs_valid << anns[i].filename << "," << anns[i].bbox.pos.x << "," << anns[i].bbox.pos.y << "," << anns[i].bbox.pos.width << "," << anns[i].bbox.pos.height << "," << anns[i].headpose.x << "," << anns[i].headpose.y << "," << anns[i].headpose.z << std::endl;
  ofs_train.close();
  ofs_valid.close();

  // Training CNN model
  UPM_PRINT("Training head-pose model");
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceHeadPoseCnn::load()
{
  // Loading head-pose
  UPM_PRINT("Loading head-pose");
  std::string deploy_file   = _data_path + "GoogLeNet_test.prototxt";
  std::string trained_model = _data_path + "GoogLeNet.caffemodel";
  try
  {
    _net = cv::dnn::readNetFromCaffe(deploy_file, trained_model);
  }
  catch (cv::Exception &ex)
  {
    UPM_ERROR("Exception: " << ex.what());
  }
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceHeadPoseCnn::process
  (
  cv::Mat frame,
  std::vector<upm::FaceAnnotation> &faces,
  const upm::FaceAnnotation &ann
  )
{
  cv::Size face_size = cv::Size(224,224);

  // Analyze each detected face
  for (FaceAnnotation &face : faces)
  {
    // Scale image
    cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -face.bbox.pos.x, 0, 1, -face.bbox.pos.y);
    cv::warpAffine(frame, face_translated, T, frame.size());
    cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << face_size.width/face.bbox.pos.width, 0, 0, 0, face_size.height/face.bbox.pos.height, 0);
    cv::warpAffine(face_translated, face_scaled, S, face_size);

    // Estimate head-pose using a CNN
    double scale_factor = 1.0;
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    cv::Mat input_blob = cv::dnn::blobFromImage(face_scaled, scale_factor, face_size, mean, swapRB);
    _net.setInput(input_blob, "data");
    cv::Mat prob = _net.forward();
    const float *output = prob.ptr<float>();

    // Store continuous head-pose
    face.headpose = cv::Point3f(output[0], output[1], output[2]);
  }
};

} // namespace upm
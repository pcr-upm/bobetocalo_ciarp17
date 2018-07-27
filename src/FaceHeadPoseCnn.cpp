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

const std::vector<std::string> MODELS = {"GoogLeNet","ResNet_50","ResNet_101","ResNet_152","VGG_16","VGG_19"};
const float BBOX_SCALE = 0.3f;
const cv::Size FACE_SIZE = cv::Size(224,224);

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
  _path = path;
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
  std::string usage = "Select model ["+MODELS[0]+", "+MODELS[1]+", "+MODELS[2]+", "+MODELS[3]+", "+MODELS[4]+", "+MODELS[5]+", "+MODELS[6]+"]";
  desc.add_options()
    ("cnn", po::value<std::string>(), usage.c_str());
  UPM_PRINT(desc);

  // Process the command line parameters
  po::variables_map vm;
  po::command_line_parser parser(argc, argv);
  parser.options(desc);
  const po::parsed_options parsed_opt(parser.allow_unregistered().run());
  po::store(parsed_opt, vm);
  po::notify(vm);

  if (vm.count("cnn"))
    _model = vm["cnn"].as<std::string>();
  if (std::find(MODELS.begin(),MODELS.end(),_model) == MODELS.end())
    throw std::invalid_argument("invalid model argument");
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
  const std::vector<upm::FaceAnnotation> &anns_train,
  const std::vector<upm::FaceAnnotation> &anns_valid
  )
{
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
  std::string deploy_file   = _path + _model + "_test.prototxt";
  std::string trained_model = _path + _model + ".caffemodel";
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
  // Analyze each detected face
  for (FaceAnnotation &face : faces)
  {
    /// Enlarge square bounding box
    cv::Point2f shift(face.bbox.pos.width*BBOX_SCALE, face.bbox.pos.height*BBOX_SCALE);
    cv::Rect_<float> bbox_enlarged = cv::Rect_<float>(face.bbox.pos.x-shift.x, face.bbox.pos.y-shift.y, face.bbox.pos.width+(shift.x*2), face.bbox.pos.height+(shift.y*2));
    /// Squared bbox required by neural networks
    bbox_enlarged.x = bbox_enlarged.x+(bbox_enlarged.width*0.5f)-(bbox_enlarged.height*0.5f);
    bbox_enlarged.width = bbox_enlarged.height;
    /// Scale image
    cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -bbox_enlarged.x, 0, 1, -bbox_enlarged.y);
    cv::warpAffine(frame, face_translated, T, bbox_enlarged.size());
    cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << FACE_SIZE.width/bbox_enlarged.width, 0, 0, 0, FACE_SIZE.height/bbox_enlarged.height, 0);
    cv::warpAffine(face_translated, face_scaled, S, FACE_SIZE);

    /// Estimate head-pose using a CNN
    double scale_factor = 1.0;
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    cv::Mat input_blob = cv::dnn::blobFromImage(face_scaled, scale_factor, FACE_SIZE, mean, swapRB);
    _net.setInput(input_blob, "data");
    cv::Mat prob = _net.forward();
    const float *output = prob.ptr<float>();

    /// Store continuous head-pose
    face.headpose = cv::Point3f(output[0], output[1], output[2]);
  }
};

} // namespace upm
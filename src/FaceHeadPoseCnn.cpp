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
#include <caffe/sgd_solvers.hpp>

namespace upm {

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
  _face_size = cv::Size(224,224);
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
  UPM_PRINT("Training head-pose");
  std::string solver_file   = _data_path + "solver.prototxt";
  std::string trained_model = _data_path + "bvlc_googlenet.caffemodel";
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::SolverParameter solver_params;
  caffe::ReadProtoFromTextFileOrDie(solver_file, &solver_params);
  caffe::Solver<float> *solver = new caffe::NesterovSolver<float>(solver_params);
  solver->net()->CopyTrainedLayersFrom(trained_model);
  solver->Solve();
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
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  _net.reset(new caffe::Net<float>(deploy_file, caffe::TEST));
  _net->CopyTrainedLayersFrom(trained_model);
  caffe::Blob<float> *input_layer = _net->input_blobs()[0];
  input_layer->Reshape(1, input_layer->channels(), input_layer->height(), input_layer->width());
  _net->Reshape();
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
    // Scale image
    cv::Mat face_scaled;
    cv::Rect bbox = intersection(face.bbox.pos, cv::Rect(0,0,frame.cols,frame.rows));
    cv::resize(frame(bbox), face_scaled, _face_size, 0, 0, cv::INTER_AREA);

    // Estimate head-pose using a CNN
    cv::Mat face_aux, face_normalized;
    face_scaled.convertTo(face_aux, CV_32FC3);
    cv::Scalar mean_pixel = cv::Scalar(133.23217514, 108.31150523, 95.0426278);
    cv::subtract(face_aux, mean_pixel, face_normalized);

    caffe::Blob<float> *input_layer = _net->input_blobs()[0];
    float *input_data = input_layer->mutable_cpu_data();
    std::vector<cv::Mat> input_channels(input_layer->channels()); // B, G, R
    for (cv::Mat &input_channel : input_channels)
    {
      input_channel = cv::Mat(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
      input_data += input_layer->height() * input_layer->width();
    }
    cv::split(face_normalized, input_channels);
    _net->Forward();

    caffe::Blob<float> *output_layer = _net->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->channels();
    std::vector<float> output = std::vector<float>(begin, end);

    // Store continuous head-pose
    face.headpose = cv::Point3f(-output[0], output[1], -output[2]);
  }
};

} // namespace upm

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
#include <H5Cpp.h>

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
  std::ofstream ofs_train(_data_path + "train_data.txt");
  std::ofstream ofs_valid(_data_path + "valid_data.txt");
  const float TRAIN_IMAGES_PERCENTAGE = 0.9f;
  const unsigned int num_data = static_cast<unsigned int>(anns.size());
  int num_train_data = static_cast<int>(num_data * TRAIN_IMAGES_PERCENTAGE);
  int num_valid_data = num_data - num_train_data;
  const float MAX_IMAGES_PER_H5_FILE = 2000;
  unsigned int num_train_sets = static_cast<unsigned int>(std::ceil(num_train_data/MAX_IMAGES_PER_H5_FILE));
  unsigned int num_valid_sets = static_cast<unsigned int>(std::ceil(num_valid_data/MAX_IMAGES_PER_H5_FILE));
  std::vector< std::vector<int> > train_sets(num_train_sets), valid_sets(num_valid_sets);
  for (int i=0; i < num_train_data; i++)
    train_sets[static_cast<unsigned int>(std::floor(i/MAX_IMAGES_PER_H5_FILE))].push_back(i);
  for (int i=0; i < num_valid_data; i++)
    valid_sets[static_cast<unsigned int>(std::floor(i/MAX_IMAGES_PER_H5_FILE))].push_back(num_train_data+i);

  // Create HDF5 databases
  for (unsigned int i=0; i < num_train_sets; i++)
  {
    char h5_path[200];
    sprintf(h5_path, "%strain_%02d.h5", _data_path.c_str(), i);
    create_HDF5_database(anns, train_sets[i], h5_path);
    ofs_train << h5_path << std::endl;
  }
  for (unsigned int i=0; i < num_valid_sets; i++)
  {
    char h5_path[200];
    sprintf(h5_path, "%svalid_%02d.h5", _data_path.c_str(), i);
    create_HDF5_database(anns, valid_sets[i], h5_path);
    ofs_valid << h5_path << std::endl;
  }
  ofs_train.close();
  ofs_valid.close();

  // Training CNN model
  UPM_PRINT("Training head-pose model");
  std::string solver_file   = _data_path + "solver.prototxt";
  std::string trained_model = _data_path + "bvlc_googlenet.caffemodel";
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  caffe::SolverParameter solver_params;
  caffe::ReadProtoFromTextFileOrDie(solver_file, &solver_params);
  boost::shared_ptr< caffe::Solver<float> > solver;
  solver.reset(new caffe::NesterovSolver<float>(solver_params));
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
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
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
  caffe::Blob<float> *input_layer = _net->input_blobs()[0];
  float *input_data = input_layer->mutable_cpu_data();
  std::vector<cv::Mat> input_channels(static_cast<unsigned int>(input_layer->channels())); // [B, G, R]
  for (cv::Mat &input_channel : input_channels)
  {
    input_channel = cv::Mat(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
    input_data += input_layer->height() * input_layer->width();
  }
  cv::Size face_size = cv::Size(_net->input_blobs()[0]->shape()[2],_net->input_blobs()[0]->shape()[3]);

  // Analyze each detected face
  for (FaceAnnotation &face : faces)
  {
    // Scale image
    cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -face.bbox.pos.x, 0, 1, -face.bbox.pos.y);
    cv::warpAffine(frame, face_translated, T, frame.size());
    cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << face_size.width/face.bbox.pos.width, 0, 0, 0, face_size.height/face.bbox.pos.height, 0);
    cv::warpAffine(face_translated, face_scaled, S, face_size);

    // Estimate head-pose using a CNN
    cv::Mat face_normalized;
    face_scaled.convertTo(face_normalized, CV_32FC3);
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
FaceHeadPoseCnn::create_HDF5_database
  (
  const std::vector<upm::FaceAnnotation> &anns,
  const std::vector<int> &anns_idx,
  std::string filename
  )
{
  cv::Size face_size = cv::Size(224,224);
  unsigned int num_indices = static_cast<unsigned int>(anns_idx.size());
  num_indices = 10;
  float label[num_indices][3], image[num_indices][3][face_size.height][face_size.width];
  boost::progress_display show_progress(num_indices);
  for (int i=0; i < num_indices; i++, ++show_progress)
  {
    upm::FaceAnnotation ann = anns[anns_idx[i]];
    cv::Mat frame = cv::imread(ann.filename, cv::IMREAD_COLOR);
    cv::Mat face_translated, T = (cv::Mat_<float>(2,3) << 1, 0, -ann.bbox.pos.x, 0, 1, -ann.bbox.pos.y);
    cv::warpAffine(frame, face_translated, T, frame.size());
    cv::Mat face_scaled, S = (cv::Mat_<float>(2,3) << face_size.width/ann.bbox.pos.width, 0, 0, 0, face_size.height/ann.bbox.pos.height, 0);
    cv::warpAffine(face_translated, face_scaled, S, face_size);
    cv::Mat face_normalized;
    face_scaled.convertTo(face_normalized, CV_32FC3);

    std::vector<cv::Mat> input_channels(static_cast<unsigned int>(face_normalized.channels())); // [B, G, R]
    cv::split(face_normalized, input_channels);
    for (unsigned int channel=0; channel < 3; channel++)
      for (unsigned int row=0; row < face_size.height; row++)
        for (unsigned int col=0; col < face_size.width; col++)
          image[i][channel][row][col] = input_channels[channel].ptr<float>(row)[col];
    label[i][0] = anns[i].headpose.x;
    label[i][1] = anns[i].headpose.y;
    label[i][2] = anns[i].headpose.z;
  }
  H5::H5File h5_ofs(filename, H5F_ACC_TRUNC);
  hsize_t imagedim[4] = {num_indices, 3, static_cast<hsize_t>(face_size.height), static_cast<hsize_t>(face_size.width)};
  H5::DataSpace image_space = H5::DataSpace(4, imagedim);
  H5::DataSet imageset = H5::DataSet(h5_ofs.createDataSet("data", H5::PredType::NATIVE_FLOAT, image_space));
  imageset.write(image, H5::PredType::NATIVE_FLOAT);
  hsize_t labeldim[2] = {num_indices, 3};
  H5::DataSpace label_space = H5::DataSpace(2, labeldim);
  H5::DataSet labelset = H5::DataSet(h5_ofs.createDataSet("label", H5::PredType::NATIVE_FLOAT, label_space));
  labelset.write(label, H5::PredType::NATIVE_FLOAT);
  h5_ofs.close();
};

} // namespace upm
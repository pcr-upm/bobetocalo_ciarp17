/** ****************************************************************************
 *  @file    FaceHeadPoseCnn.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2016/01
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_HEADPOSE_CNN_HPP
#define FACE_HEADPOSE_CNN_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceHeadPose.hpp>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>
#include <caffe/caffe.hpp>

namespace upm {

/** ****************************************************************************
 * @class FaceHeadPoseCnn
 * @brief Class used for head pose estimation.
 ******************************************************************************/
class FaceHeadPoseCnn: public FaceHeadPose
{
public:
  FaceHeadPoseCnn(std::string path);

  ~FaceHeadPoseCnn() {};

  void
  parseOptions
    (
    int argc,
    char **argv
    );

  void
  train
    (
    const std::vector<upm::FaceAnnotation> &anns
    );

  void
  load();

  void
  process
    (
    cv::Mat frame,
    std::vector<upm::FaceAnnotation> &faces,
    const upm::FaceAnnotation &ann
    );

  void
  create_HDF5_database
    (
    const std::vector<upm::FaceAnnotation> &anns,
    const std::vector<int> &anns_idx,
    std::string filename
    );

  std::string _data_path;
  boost::shared_ptr< caffe::Net<float> > _net;
};

} // namespace upm

#endif /* FACE_HEADPOSE_CNN_HPP */

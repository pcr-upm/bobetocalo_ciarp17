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
#include <opencv2/dnn.hpp>

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
    const std::vector<upm::FaceAnnotation> &anns_train,
    const std::vector<upm::FaceAnnotation> &anns_valid
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

  std::string _data_path;
  cv::dnn::Net _net;
};

} // namespace upm

#endif /* FACE_HEADPOSE_CNN_HPP */

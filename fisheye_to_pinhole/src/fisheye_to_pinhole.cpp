/*
 * Created on Sat Oct 12 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

// STL
#include <iostream>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
// Yaml
#include <yaml-cpp/yaml.h>

using namespace std;
using namespace cv;

struct PolyfisheyeIntrins {
  double A11;
  double A12;
  double A22;
  double u0;
  double v0;
  double k_2_7[6];  // k2-k7
};

/**
 * @brief Project pinhole's pixel pt into fisheye's pixel pt.
 *
 * @param pt virtual pinhole camera's pixel point
 * @param pinhole_k pinhole's intrins
 * @param R pinhole's rotation w.r.t fisheye camera.
 * @param t pinhole's translation w.r.t fisheye camera.
 * @param intrins fisheye's intrins
 * @return Eigen::Vector2d fisheye camera's pixel pt
 */
Eigen::Vector2d ProjPinholeToFisheye(Eigen::Vector2d pt, Eigen::Matrix3d pinhole_k, Eigen::Matrix3d R,
                                     Eigen::Vector3d t, PolyfisheyeIntrins &intrins) {
  // Back project pinhole points.
  double pinhole_fx = pinhole_k(0, 0);
  double pinhole_fy = pinhole_k(1, 1);
  double pinhole_u0 = pinhole_k(0, 2);
  double pinhole_v0 = pinhole_k(1, 2);
  // Get points at carema image plane coordinate
  Eigen::Vector3d obj_pt;
  obj_pt.x() = (pt.x() - pinhole_u0) / pinhole_fx;
  obj_pt.y() = (pt.y() - pinhole_v0) / pinhole_fy;
  obj_pt.z() = 1;
  obj_pt = R * obj_pt;
  // Project points into fisheye camera.
  double theta = acos(obj_pt(2) / obj_pt.norm());
  double inverse_r_P2 = 1.0 / sqrt(obj_pt(1) * obj_pt(1) + obj_pt(0) * obj_pt(0));
  double sin_phi = obj_pt(1) * inverse_r_P2;
  double cos_phi = obj_pt(0) * inverse_r_P2;

  double R0_theta = theta;
  double thetas = theta * theta;
  for (int i = 2; i <= 7; i++) {
    R0_theta += intrins.k_2_7[i - 2] * thetas;
    thetas *= theta;
  }
  Eigen::Matrix2d Af = Eigen::Matrix2d::Zero();
  Af(0, 0) = intrins.A11;
  Af(0, 1) = intrins.A12;
  Af(1, 1) = intrins.A22;
  Eigen::Vector2d u = R0_theta * Af * Eigen::Vector2d(cos_phi, sin_phi) + Eigen::Vector2d(intrins.u0, intrins.v0);
  return u;
}

// Since it's not a good habit to use global variables, it's convenient for a simple node use.

ros::Publisher left_img_pub, right_img_pub, left_cam_info_pub, right_cam_info_pub;
sensor_msgs::CameraInfo left_cam_info, right_cam_info;

void Callback(const sensor_msgs::Image::ConstPtr &img_ptr) {
  cv_bridge::CvImageConstPtr ros_img = cv_bridge::toCvShare(img_ptr);
  Mat left_img, right_img;
  remap(ros_img->image, left_img, left_mapX, left_mapY, INTER_LINEAR);
  remap(ros_img->image, right_img, right_mapX, right_mapY, INTER_LINEAR);

  sensor_msgs::ImagePtr left_img_msg = cv_bridge::CvImage(img_ptr->header, "mono8", left_img).toImageMsg();
  sensor_msgs::ImagePtr right_img_msg = cv_bridge::CvImage(img_ptr->header, "mono8", right_img).toImageMsg();

  left_cam_info.header = img_ptr->header;
  right_cam_info.header = img_ptr->header;

  left_img_pub.publish(left_img_msg);
  left_cam_info_pub.publish(left_cam_info);
  right_img_pub.publish(right_img_msg);
  right_cam_info_pub.publish(right_cam_info);
}

int main(int argc, char **argv) {
  std::string config_file = argv[1];
  YAML::Node config = YAML::LoadFile(config_file);
  if (config.IsNull()) {
    system("pause");
    return -1;
  }
  // Get parameters.
  PolyfisheyeIntrins intrins;
  int pinhole_image_width, pinhole_image_height;

  std::vector<double> fisheye_camera_T_data;

  intrins.A11 = config["projection_parameters/A11"].as<double>();
  intrins.A12 = config["projection_parameters/A11"].as<double>();
  intrins.A22 = config["projection_parameters/A11"].as<double>();

  intrins.k_2_7[0] = config["projection_parameters/k2"].as<double>();
  intrins.k_2_7[1] = config["projection_parameters/k2"].as<double>();
  intrins.k_2_7[2] = config["projection_parameters/k2"].as<double>();
  intrins.k_2_7[3] = config["projection_parameters/k2"].as<double>();
  intrins.k_2_7[4] = config["projection_parameters/k2"].as<double>();
  intrins.k_2_7[5] = config["projection_parameters/k2"].as<double>();

  intrins.u0 = config["projection_parameters/u0"].as<double>();
  intrins.v0 = config["projection_parameters/v0"].as<double>();

  pinhole_image_width = config["pinhole_image_width"].as<double>();
  pinhole_image_height = config["pinhole_image_height"].as<double>();

  fisheye_camera_T_data = config["fisheye_camera_T"].as<double>();

  Eigen::Matrix4d fisheye_camera_T = Eigen::Matrix4d::Zero();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      fisheye_camera_T(i, j) = fisheye_camera_T_data[i * 4 + j];
    }
  }
  fisheye_camera_T(3, 3) = 1;

  // Set virtual camera information
  double pinhole_cx, pinhole_cy;
  pinhole_cx = pinhole_cy = pinhole_image_width / 2;

  Eigen::Matrix3d pinhole_k = Eigen::Matrix3d::Zero();
  pinhole_k(0, 0) = pinhole_cx;
  pinhole_k(1, 1) = pinhole_cy;
  pinhole_k(0, 2) = pinhole_image_width / 2;
  pinhole_k(1, 2) = pinhole_image_height / 2;
  pinhole_k(2, 2) = 1;

  left_mapX = Mat::zeros(pinhole_image_height, pinhole_image_width, CV_32F);
  left_mapY = Mat::zeros(pinhole_image_height, pinhole_image_width, CV_32F);
  right_mapX = Mat::zeros(pinhole_image_height, pinhole_image_width, CV_32F);
  right_mapY = Mat::zeros(pinhole_image_height, pinhole_image_width, CV_32F);

  Eigen::AngleAxisd left_angle(M_PI / 4, Eigen::Vector3d(0, -1, 0));
  Eigen::Matrix3d left_R = left_angle.matrix();
  cout << left_R << endl;
  Eigen::AngleAxisd right_angle(M_PI / 4, Eigen::Vector3d(0, 1, 0));
  Eigen::Matrix3d right_R = right_angle.matrix();

  left_cam_info.height = pinhole_image_height;
  left_cam_info.width = pinhole_image_width;
  for (int i = 0; i < 9; i++)
    left_cam_info.K[i] = pinhole_k(i);
  left_cam_info.D = vector<double>(4, 0);  // No distortion
  Eigen::MatrixXd left_T(3, 4), left_proj(3, 4);

  left_T.block(0, 0, 3, 3) = left_R * fisheye_camera_T.block(0, 0, 3, 3);
  left_T.block(0, 3, 3, 1) = fisheye_camera_T.block(0, 3, 3, 1);
  left_proj = pinhole_k * left_T;  // P = K * T
  for (int i = 0; i < 9; i++)
    left_cam_info.R[i] = left_T(i);
  for (int i = 0; i < 12; i++)
    left_cam_info.P[i] = left_proj(i);

  right_cam_info.height = pinhole_image_height;
  right_cam_info.width = pinhole_image_width;
  for (int i = 0; i < 9; i++)
    right_cam_info.K[i] = pinhole_k(i);
  right_cam_info.D = vector<double>(4, 0);  // No distortion

  Eigen::MatrixXd right_T(3, 4), right_proj(3, 4);
  right_T.block(0, 0, 3, 3) = right_R * fisheye_camera_T.block(0, 0, 3, 3);
  right_T.block(0, 3, 3, 1) = fisheye_camera_T.block(0, 3, 3, 1);
  right_proj = pinhole_k * right_T;  // P = K * T
  for (int i = 0; i < 9; i++)
    right_cam_info.R[i] = right_T(i);
  for (int i = 0; i < 12; i++)
    right_cam_info.P[i] = right_proj(i);

  // Start construct remapping relationship.
  for (int i = 0; i < pinhole_image_height; i++)
    for (int j = 0; j < pinhole_image_width; j++) {
      Eigen::Vector2d left_map_pt =
          ProjPinholeToFisheye(Eigen::Vector2d(j, i), pinhole_k, left_R, Eigen::Vector3d::Zero(), intrins);

      left_mapX.at<float>(i, j) = left_map_pt.x();
      left_mapY.at<float>(i, j) = left_map_pt.y();

      Eigen::Vector2d right_map_pt =
          ProjPinholeToFisheye(Eigen::Vector2d(j, i), pinhole_k, right_R, Eigen::Vector3d::Zero(), intrins);

      right_mapX.at<float>(i, j) = right_map_pt.x();
      right_mapY.at<float>(i, j) = right_map_pt.y();
    }

  return 0;
}
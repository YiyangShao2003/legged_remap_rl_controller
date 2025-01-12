//
// Created by qiayuanl on 9/1/24.
//

#pragma once
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <realtime_tools/realtime_box.h>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "legged_controllers/ControllerBase.h"

namespace legged {
class LeggedRemapRlController : public ControllerBase {
  using tensor_element_t = float;
  using Twist = geometry_msgs::msg::TwistStamped;

 public:
  controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 protected:
  virtual vector_t getObservations(const rclcpp::Time& time);
  virtual vector_t playModel(const vector_t& observations) const;
  virtual vector_t remapJointOrder(const vector_t & policy_vector) const;

  // Onnx
  std::shared_ptr<Ort::Env> onnxEnvPrt_;
  std::unique_ptr<Ort::Session> sessionPtr_;
  std::vector<const char*> inputNames_;
  std::vector<const char*> outputNames_;
  std::vector<std::vector<int64_t>> inputShapes_;
  std::vector<std::vector<int64_t>> outputShapes_;

  vector_t lastActions_;
  std::vector<std::string> jointNameInPolicy_;

  // Observation
  std::vector<std::string> observationNames_;
  size_t observationSize_{0};
  std::vector<size_t> obsIndexMap_;

  // Phase
  vector_t phase_;
  scalar_t phaseDt_{2 * M_PI * 0.02 * 1.5};

  vector3_t command_;

  // Action
  scalar_t actionScale_{0};
  std::string actionType_{"position_absolute"};
  vector_t desiredPosition_;

  // Time
  bool firstUpdate_{false};
  scalar_t policyFrequency_{50.};
  rclcpp::Time lastPlayTime_;

  // Command Interface
  rclcpp::Subscription<Twist>::SharedPtr velocitySubscriber_;
  realtime_tools::RealtimeBox<std::shared_ptr<Twist>> receivedVelocityMsg_;

  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>> publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>> publisherRealtime_;
};

}  // namespace legged

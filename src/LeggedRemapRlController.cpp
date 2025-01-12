//
// Created by qiayuanl on 9/1/24.
//

#include <memory>

#include "legged_remap_rl_controller/LeggedRemapRlController.h"

namespace legged {

void printInputsOutputs(const std::vector<const char*>& inputNames, const std::vector<std::vector<int64_t>>& inputShapes,
                        const std::vector<const char*>& outputNames, const std::vector<std::vector<int64_t>>& outputShapes) {
  std::cout << "Inputs:" << std::endl;
  for (size_t i = 0; i < inputNames.size(); ++i) {
    std::cout << "  Input " << i << ": " << inputNames[i] << " - Shape: [";
    for (size_t j = 0; j < inputShapes[i].size(); ++j) {
      std::cout << inputShapes[i][j];
      if (j < inputShapes[i].size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  std::cout << "Outputs:" << std::endl;
  for (size_t i = 0; i < outputNames.size(); ++i) {
    std::cout << "  Output " << i << ": " << outputNames[i] << " - Shape: [";
    for (size_t j = 0; j < outputShapes[i].size(); ++j) {
      std::cout << outputShapes[i][j];
      if (j < outputShapes[i].size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
}

controller_interface::return_type LeggedRemapRlController::update(const rclcpp::Time& time, const rclcpp::Duration& period) {
  if (ControllerBase::update(time, period) != controller_interface::return_type::OK) {
    return controller_interface::return_type::ERROR;
  }

  std::shared_ptr<Twist> lastCommandMsg;
  receivedVelocityMsg_.get(lastCommandMsg);
  if (time - lastCommandMsg->header.stamp > std::chrono::milliseconds{static_cast<int>(0.5 * 1000.0)}) {
    lastCommandMsg->twist.linear.x = 0.0;
    lastCommandMsg->twist.linear.y = 0.0;
    lastCommandMsg->twist.angular.z = 0.0;
  }
  command_ << lastCommandMsg->twist.linear.x, lastCommandMsg->twist.linear.y, lastCommandMsg->twist.angular.z;

  if (firstUpdate_ || (time - lastPlayTime_).seconds() >= 1. / policyFrequency_) {
    const vector_t observations = getObservations(time);
    lastActions_ = playModel(observations);

    if (actionType_ == "position_absolute") {
      for (size_t i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] = lastActions_[i] * actionScale_ + defaultPosition_[hardwareIndex];
      }
    } else if (actionType_ == "position_relative") {
      for (size_t i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] += lastActions_[i] * actionScale_;
      }
    } else if (actionType_ == "position_delta") {
      const vector_t currentPosition = leggedModel_->getLeggedModel()->getGeneralizedPosition().tail(lastActions_.size());
      for (size_t i = 0; i < lastActions_.size(); ++i) {
        size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
        desiredPosition_[hardwareIndex] = currentPosition[hardwareIndex] + lastActions_[i] * actionScale_;
      }
    }
    setPositions(desiredPosition_);

    firstUpdate_ = false;
    lastPlayTime_ = time;

    if (publisherRealtime_->trylock()) {
      auto& msg = publisherRealtime_->msg_;
      msg.data.clear();
      for (const double i : observations) {
        msg.data.push_back(i);
      }
      for (const double i : lastActions_) {
        msg.data.push_back(i);
      }
      publisherRealtime_->unlockAndPublish();
    }
  }

  return controller_interface::return_type::OK;
}

controller_interface::CallbackReturn LeggedRemapRlController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_configure(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  // Onnx
  std::string policyPath{};
  get_node()->get_parameter("policy.path", policyPath);
  get_node()->get_parameter("policy.frequency", policyFrequency_);
  onnxEnvPrt_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LeggedRemapRlController");
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetInterOpNumThreads(1);
  sessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyPath.c_str(), sessionOptions);
  inputNames_.clear();
  outputNames_.clear();
  inputShapes_.clear();
  outputShapes_.clear();
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < sessionPtr_->GetInputCount(); i++) {
    inputNames_.push_back(sessionPtr_->GetInputName(i, allocator));
    inputShapes_.push_back(sessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < sessionPtr_->GetOutputCount(); i++) {
    outputNames_.push_back(sessionPtr_->GetOutputName(i, allocator));
    outputShapes_.push_back(sessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  printInputsOutputs(inputNames_, inputShapes_, outputNames_, outputShapes_);

  const size_t numJoints = leggedModel_->getLeggedModel()->getJointNames().size();

  jointNameInPolicy_ = get_node()->get_parameter("policy.joint_names").as_string_array();
  if (jointNameInPolicy_.size() != numJoints) {
    RCLCPP_ERROR(get_node()->get_logger(), "joint_names size is not equal to joint size.");
    return controller_interface::CallbackReturn::ERROR;
  }
  auto jointNameObsOrder_ = get_node()->get_parameter("policy.obs_order_joint_names").as_string_array();
  if (jointNameObsOrder_.size() != jointNameInPolicy_.size()) {
    RCLCPP_ERROR(get_node()->get_logger(), "obs_order_joint_names size (%zu) does not match policy joint_names size (%zu).",
                  jointNameObsOrder_.size(), jointNameInPolicy_.size());
    return controller_interface::CallbackReturn::ERROR;
  }
  std::unordered_map<std::string, size_t> policyIndexMap_;
  for (size_t i = 0; i < jointNameInPolicy_.size(); ++i) {
    policyIndexMap_[jointNameInPolicy_[i]] = i;
  }
  obsIndexMap_.reserve(jointNameObsOrder_.size());
  for (const auto & joint_name : jointNameObsOrder_) {
    auto it = policyIndexMap_.find(joint_name);
    if (it != policyIndexMap_.end()) {
      obsIndexMap_.push_back(it->second);
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Joint name '%s' in obs_order_joint_names not found in policy joint_names.",
                    joint_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }

  lastActions_.setZero(numJoints);
  command_.setZero();
  RCLCPP_INFO_STREAM(rclcpp::get_logger("LeggedRemapRlController"), "Load Onnx model from" << policyPath << " successfully !");

  // Observation
  observationNames_ = get_node()->get_parameter("policy.observations").as_string_array();
  observationSize_ = 0;
  for (const auto& name : observationNames_) {
    if (name == "base_lin_vel") {
      observationSize_ += 3;
    } else if (name == "base_ang_vel") {
      observationSize_ += 3;
    } else if (name == "projected_gravity") {
      observationSize_ += 3;
    } else if (name == "joint_positions") {
      observationSize_ += numJoints;
    } else if (name == "joint_velocities") {
      observationSize_ += numJoints;
    } else if (name == "last_action") {
      observationSize_ += numJoints;
    } else if (name == "phase") {
      observationSize_ += 4;
    } else if (name == "command") {
      observationSize_ += 3;
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Unknown observation name: %s", name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }
  phase_.setZero(2);
  phase_[1] = M_PI;

  // Action
  get_node()->get_parameter("policy.action_scale", actionScale_);
  get_node()->get_parameter("policy.action_type", actionType_);
  if (actionType_ != "position_absolute" && actionType_ != "position_relative" && actionType_ != "position_delta") {
    RCLCPP_ERROR(get_node()->get_logger(), "Unknown action type: %s", actionType_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  desiredPosition_.setZero(numJoints);

  // ROS Interface
  velocitySubscriber_ =
      get_node()->create_subscription<Twist>("/cmd_vel", rclcpp::SystemDefaultsQoS(), [this](const std::shared_ptr<Twist> msg) -> void {
        if ((msg->header.stamp.sec == 0) && (msg->header.stamp.nanosec == 0)) {
          RCLCPP_WARN_ONCE(get_node()->get_logger(),
                           "Received TwistStamped with zero timestamp, setting it to current "
                           "time, this message will only be shown once");
          msg->header.stamp = get_node()->get_clock()->now();
        }
        receivedVelocityMsg_.set(msg);
      });

  publisher_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>("~/policy_io", rclcpp::SystemDefaultsQoS());
  publisherRealtime_ = std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>(publisher_);

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn LeggedRemapRlController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (ControllerBase::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }
  firstUpdate_ = true;
  receivedVelocityMsg_.set(std::make_shared<Twist>());
  lastActions_.setZero();
  desiredPosition_ = defaultPosition_;
  return controller_interface::CallbackReturn::SUCCESS;
}

vector_t LeggedRemapRlController::getObservations(const rclcpp::Time& time) {
  const auto leggedModel = leggedModel_->getLeggedModel();
  const auto q = leggedModel->getGeneralizedPosition();
  const auto v = leggedModel->getGeneralizedVelocity();

  const quaternion_t quat(q.segment<4>(3));
  const matrix_t inverseRot = quat.toRotationMatrix().transpose();
  const vector3_t baseLinVel = inverseRot * v.segment<3>(0);
  const auto& angVelArray = imu_->get_angular_velocity();
  const vector3_t baseAngVel(angVelArray[0], angVelArray[1], angVelArray[2]);
  const vector3_t gravityVector(0, 0, -1);
  const vector3_t projectedGravity(inverseRot * gravityVector);

  const vector_t jointPositions = q.tail(lastActions_.size());
  const vector_t jointVelocities = v.tail(lastActions_.size());

  vector_t jointPositionsInPolicy(jointNameInPolicy_.size());
  vector_t jointVelocitiesInPolicy(jointNameInPolicy_.size());
  for (size_t i = 0; i < jointNameInPolicy_.size(); ++i) {
    const size_t hardwareIndex = jointIndexMap_[jointNameInPolicy_[i]];
    jointPositionsInPolicy[i] = jointPositions[hardwareIndex] - defaultPosition_[hardwareIndex];
    jointVelocitiesInPolicy[i] = jointVelocities[hardwareIndex];
  }

  vector_t observations(observationSize_);
  size_t index = 0;
  for (const auto name : observationNames_) {
    if (name == "base_lin_vel") {
      observations.segment<3>(index) = baseLinVel;
      index += 3;
    } else if (name == "base_ang_vel") {
      observations.segment<3>(index) = baseAngVel;
      index += 3;
    } else if (name == "projected_gravity") {
      observations.segment<3>(index) = projectedGravity;
      index += 3;
    } else if (name == "joint_positions") {
      observations.segment(index, jointPositionsInPolicy.size()) = remapJointOrder(jointPositionsInPolicy);
      index += jointPositionsInPolicy.size();
    } else if (name == "joint_velocities") {
      observations.segment(index, jointVelocitiesInPolicy.size()) = remapJointOrder(jointVelocitiesInPolicy);
      index += jointVelocitiesInPolicy.size();
    } else if (name == "last_action") {
      observations.segment(index, lastActions_.size()) = lastActions_;
      index += lastActions_.size();
    } else if (name == "phase") {
      vector_t phase(4);
      phase[0] = std::cos(phase_[0]);
      phase[1] = std::cos(phase_[1]);
      phase[2] = std::sin(phase_[0]);
      phase[3] = std::sin(phase_[1]);
      phase_[0] = std::fmod(phase_[0] + phaseDt_ + M_PI, 2 * M_PI) - M_PI;
      phase_[1] = std::fmod(phase_[1] + phaseDt_ + M_PI, 2 * M_PI) - M_PI;
      observations.segment(index, phase.size()) = phase;
      index += phase.size();
    } else if (name == "command") {
      observations.segment<3>(index) = command_;
      index += 3;
    }
  }
  return observations;
}

vector_t LeggedRemapRlController::playModel(const vector_t& observations) const {
  // clang-format on
  std::vector<tensor_element_t> observationTensor;
  for (const double i : observations) {
    observationTensor.push_back(static_cast<tensor_element_t>(i));
  }
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  inputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, observationTensor.data(), observationTensor.size(),
                                                                   inputShapes_[0].data(), inputShapes_[0].size()));
  // run inference
  const Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = sessionPtr_->Run(runOptions, inputNames_.data(), inputValues.data(), 1, outputNames_.data(), 1);

  vector_t actions(lastActions_.size());
  for (size_t i = 0; i < actions.size(); ++i) {
    actions[i] = outputValues[0].At<tensor_element_t>({0, i});
  }
  return actions;
}


vector_t LeggedRemapRlController::remapJointOrder(const vector_t & raw_vector) const
  {
    if (raw_vector.size() != static_cast<long>(jointNameInPolicy_.size())) {
      throw std::runtime_error("Vector size does not match jointNameInPolicy_ size.");
    }

    vector_t obs(obsIndexMap_.size());

    for (size_t i = 0; i < obsIndexMap_.size(); ++i) {
      obs(static_cast<long>(i)) = raw_vector(static_cast<long>(obsIndexMap_[i]));
    }

    return obs;
  }

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::LeggedRemapRlController, controller_interface::ControllerInterface)

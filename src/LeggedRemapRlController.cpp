#include <memory>

#include "legged_remap_rl_controller/LeggedRemapRlController.h"

namespace legged {
vector_t LeggedRemapRlController::playModel(const vector_t& observations) const {
  std::cerr << "LeggedRemapRlController::playModel" << std::endl;
  return OnnxController::playModel(observations);
}
}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::LeggedRemapRlController, controller_interface::ControllerInterface)

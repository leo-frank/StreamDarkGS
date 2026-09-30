#!/usr/bin/env bash
# Read-only inventory for the Go2's built-in front camera calibration.
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-<CycloneDDS><Domain><General><NetworkInterfaceAddress>eth1</NetworkInterfaceAddress></General></Domain></CycloneDDS>}"

echo '[camera-check] ROS topics'
if command -v ros2 >/dev/null 2>&1; then
  timeout 10 ros2 topic list -t 2>&1 | grep -Ei 'camera|video|image|calib|intrinsic' || true
  echo '[camera-check] ROS services'
  timeout 10 ros2 service list -t 2>&1 | grep -Ei 'camera|video|calib|intrinsic' || true
  echo '[camera-check] ROS nodes'
  timeout 10 ros2 node list 2>&1 | grep -Ei 'camera|video|image' || true
else
  echo 'ros2 command unavailable'
fi

echo '[camera-check] Candidate files'
for directory in /etc /opt /usr/local/etc /home/unitree; do
  if [[ -d "$directory" ]]; then
    find "$directory" -type f \
      \( -iname '*camera*info*' -o -iname '*camera*calib*' \
      -o -iname '*intrinsic*' -o -iname '*distort*' \
      -o -iname '*front*camera*.yaml' -o -iname '*front*camera*.json' \) \
      -print 2>/dev/null
  fi
done | head -100

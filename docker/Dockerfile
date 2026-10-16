FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Gazebo Fortress / Sim and ROS 2 Humble integration packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ignition-fortress \
    ros-humble-ros-gz \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-image \
    ros-humble-ros-gz-interfaces \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-nav-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    python3-colcon-common-extensions \
    python3-pip \
    python3-matplotlib \
    python3-numpy \
    python3-yaml \
    python3-pil \
    python3-opencv \
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    x11-utils \
    procps \
    net-tools \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV IGN_GAZEBO_RESOURCE_PATH=/workspace/ros2_ws/src/swarm_gazebo/models:/workspace/ros2_ws/src/swarm_gazebo/worlds
ENV PYTHONPATH=/workspace:/workspace/ros2_ws/src/swarm_gazebo

WORKDIR /workspace

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "if [ -f /workspace/ros2_ws/install/setup.bash ]; then source /workspace/ros2_ws/install/setup.bash; fi" >> /root/.bashrc

CMD ["/bin/bash"]

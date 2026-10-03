FROM ghcr.io/tiryoh/ros2-desktop-vnc:humble-20260927T1000

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-gz \
        ros-humble-xacro \
        ros-humble-rqt-tf-tree \
        ros-humble-plotjuggler-ros \
        ros-humble-rosbag2-storage-mcap \
        openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y numpy \
    && echo 'numpy<2' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN printf '[org.mate.screensaver]\nidle-activation-enabled=false\nlock-enabled=false\n' \
        > /usr/share/glib-2.0/schemas/99_rover-sim.gschema.override \
    && glib-compile-schemas /usr/share/glib-2.0/schemas

RUN mkdir -p /run/sshd \
    && printf 'PasswordAuthentication yes\nPermitRootLogin no\nAllowUsers ubuntu\n' \
        > /etc/ssh/sshd_config.d/rover-sim.conf \
    && echo 'DISPLAY=:1' >> /etc/environment
COPY docker/sshd.conf /etc/supervisor/conf.d/sshd.conf
COPY --chmod=755 docker/ros-python3 /opt/ros-python/bin/python

RUN cat >> /etc/skel/.bashrc <<'BASHRC'
source /opt/ros/humble/setup.bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
[ -f /workspaces/rover-sim/install/setup.bash ] && source /workspaces/rover-sim/install/setup.bash
alias cb='(cd /workspaces/rover-sim && colcon build --symlink-install) && source /workspaces/rover-sim/install/setup.bash'
BASHRC

ENV DISPLAY=:1
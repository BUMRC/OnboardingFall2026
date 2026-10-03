#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/$ROS_DISTRO/setup.bash

start_desktop() {
    rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
    mkdir -p /tmp/desktop
    Xvnc :1 -geometry 1600x900 -depth 24 -rfbport 5901 -localhost -SecurityTypes None -AlwaysShared \
        > /tmp/desktop/xvnc.log 2>&1 &
    for _ in $(seq 50); do
        [ -S /tmp/.X11-unix/X1 ] && break
        sleep 0.1
    done
    dbus-launch --exit-with-session xfce4-session > /tmp/desktop/xfce.log 2>&1 &
    websockify --web /usr/share/novnc 6080 localhost:5901 > /tmp/desktop/novnc.log 2>&1 &
}

start_sshd() {
    {
        echo 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"'
        env | grep -E '^(DISPLAY|ROS_|RMW_|RCUTILS_|PIP_CONSTRAINT)'
    } | sudo tee /etc/environment > /dev/null
    sudo mkdir -p /run/sshd
    sudo /usr/sbin/sshd
}

start_desktop
start_sshd
exec "$@"
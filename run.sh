xhost +local:root
docker build -t seg_net_nrp docker/
docker run -d -t --net host --gpus all -v $PWD/:/workspace --shm-size=64g -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=unix$DISPLAY --device /dev/dri --privileged -v /home/$USER/.Xauthority:/root/.Xauthority seg_net_nrp
docker build -t hbp-vision .
docker run -d -t --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=unix$DISPLAY --device /dev/dri --privileged -v /home/$USER/.Xauthority:/root/.Xauthority -v $PWD:/workspace hbp-vision


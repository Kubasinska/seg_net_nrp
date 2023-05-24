FROM ubuntu:20.04

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y vim wget nano curl git git-lfs ca-certificates build-essential
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y ffmpeg libsm6 libxext6 zip htop screen libgl1-mesa-glx

SHELL ["/bin/bash", "-c"]
WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN /root/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
RUN echo 'conda activate main' >> ~/.bashrc
RUN /root/miniconda3/bin/conda create --name main python==3.9
RUN /root/miniconda3/bin/conda run -n main conda install numpy
COPY requirements.txt /root/

RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" | tee /etc/apt/sources.list.d/ros-focal.list
RUN curl http://repo.ros2.org/repos.key | apt-key add -
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install ros-noetic-desktop-full git-lfs ros-noetic-object-recognition-msgs ros-noetic-moveit-core ros-noetic-moveit-ros-perception ros-noetic-moveit-ros-planning-interface ros-noetic-velocity-controllers ros-noetic-twist-mux python3-rostopic ros-noetic-effort-controllers ros-noetic-position-controllers ros-noetic-joint-trajectory-controller ros-noetic-moveit ros-noetic-rviz-visual-tools ros-noetic-moveit-visual-tools -y
# Nano settings
RUN /root/miniconda3/bin/conda run -n main  pip install --upgrade pip
RUN /root/miniconda3/bin/conda run -n main  pip install cvbridge3
RUN /root/miniconda3/bin/conda run -n main  pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
RUN /root/miniconda3/bin/conda run -n main  pip install imageio==2.16.1
RUN /root/miniconda3/bin/conda run -n main  pip install matplotlib numpy
RUN /root/miniconda3/bin/conda run -n main  pip install opencv-python==4.5.5.64
RUN /root/miniconda3/bin/conda run -n main conda install -c conda-forge catkin_tools
RUN /root/miniconda3/bin/conda run -n main conda install -c conda-forge rosdep
RUN #/root/miniconda3/bin/conda run -n main  pip install sensor-msgs==1.13.1
RUN /root/miniconda3/bin/conda run -n main  pip install torch==1.8.1
RUN /root/miniconda3/bin/conda run -n main  pip install torchvision==0.9.1
RUN /root/miniconda3/bin/conda run -n main  pip install h5py
RUN /root/miniconda3/bin/conda run -n main  pip install albumentations



RUN #/root/miniconda3/bin/conda run -n main pip install -r /root/requirements.txt

WORKDIR /workspace

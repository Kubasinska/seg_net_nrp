FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]
WORKDIR /workspace
RUN apt-get update
RUN apt-get -y install curl gnupg2  wget git

# ROS Installation
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive"  apt-get -y install ros-noetic-ros-base ros-noetic-cv-bridge

# Dependencies and Python
WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN /root/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init

RUN /root/miniconda3/bin/conda create --name hbp python=3.7
#RUN echo 'conda activate hbp' >> ~/.bashrc
RUN /root/miniconda3/bin/conda install --name hbp pip
COPY requirements.txt requirements.txt
RUN . /opt/ros/noetic/setup.sh && /root/miniconda3/bin/conda run -n hbp pip install -r requirements.txt
RUN #echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
RUN /root/miniconda3/bin/conda run -n hbp conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN apt-get update
RUN apt-get install ros-noetic-rviz -y

#RUN /root/miniconda3/bin/conda run -n hbp pip install opencv-python
RUN /root/miniconda3/bin/conda run -n hbp conda install -y -c conda-forge ros-rospy
RUN /root/miniconda3/bin/conda run -n hbp conda install -y -c conda-forge ros-cv-bridge
RUN /root/miniconda3/bin/conda run -n hbp conda install -y -c conda-forge ros-sensor-msgs
RUN /root/miniconda3/bin/conda run -n hbp pip install ipython
#RUN echo 'conda activate hbp' >> ~/.bashrc

WORKDIR /
RUN echo 'alias python=/root/miniconda3/envs/hbp/bin/python' >> ~/.bashrc
RUN echo 'alias ipython=/root/miniconda3/envs/hbp/bin/ipython' >> ~/.bashrc
RUN echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
RUN apt-get update
RUN apt-get -y install vim

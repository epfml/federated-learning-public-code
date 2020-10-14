FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04


# install some necessary tools.
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    pkg-config \
    software-properties-common
RUN apt-get install -y \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    libcurl3-dev \
    libfreetype6-dev \
    libzmq3-dev \
    libcupti-dev \
    pkg-config \
    libav-tools \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    locales
RUN apt-get install -y \
    sudo \
    rsync \
    cmake \
    g++ \
    swig \
    vim \
    git \
    curl \
    wget \
    unzip \
    zsh \
    git \
    screen \
    tmux \
    openssh-server
RUN apt-get update && \
    apt-get install -y pciutils net-tools iputils-ping && \
    apt-get install -y htop
RUN add-apt-repository ppa:openjdk-r/ppa \
    && apt-get update \
    && apt-get install -y \
    openjdk-7-jdk \
    openjdk-7-jre-headless
# install good vim.
RUN curl http://j.mp/spf13-vim3 -L -o - | sh

# configure environments.
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# configure user.
ENV SHELL=/bin/bash \
    NB_USER=user \
    NB_UID=1000 \
    NB_GROUP=unit \
    NB_GID=1000
ENV HOME=/home/$NB_USER

ADD base/fix-permissions /usr/local/bin/fix-permissions
RUN chmod +x /usr/local/bin/fix-permissions
ADD base/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER} && \
    fix-permissions $HOME
RUN echo 'user ALL=(ALL) NOPASSWD: ALL' | sudo EDITOR='tee -a' visudo

# Default ssh config file that skips (yes/no) question when first login to the host
RUN mkdir /var/run/sshd
RUN sed -i "s/#PasswordAuthentication.*/PasswordAuthentication no/g" /etc/ssh/sshd_config \
    && sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/g" /etc/ssh/sshd_config \
    && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config \
    && sed -i "s/#AuthorizedKeysFile/AuthorizedKeysFile/g" /etc/ssh/sshd_config
RUN /usr/bin/ssh-keygen -A

ENV SSHDIR $HOME/.ssh
RUN mkdir -p $SSHDIR \
    && chmod go-w $HOME/ \
    && chmod 700 $SSHDIR \
    && touch $SSHDIR/authorized_keys \
    && chmod 600 $SSHDIR/authorized_keys \
    && chown -R ${NB_USER}:${NB_GROUP} ${SSHDIR} \
    && chown -R ${NB_USER}:${NB_GROUP} /etc/ssh/*

###### switch to user and compile test example.
USER ${NB_USER}
RUN ssh-keygen -b 2048 -t rsa -f $SSHDIR/id_rsa -q -N ""
RUN cat ${SSHDIR}/*.pub >> ${SSHDIR}/authorized_keys
RUN echo "StrictHostKeyChecking no" > ${SSHDIR}/config

# configure screen and tmux
ADD base/.tmux.conf $HOME/
ADD base/.screenrc $HOME/

# expose port for ssh and start ssh service.
EXPOSE 22
# expose port for notebook.
EXPOSE 8888
# expose port for tensorboard.
EXPOSE 6666

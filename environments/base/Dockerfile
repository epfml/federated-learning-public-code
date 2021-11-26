FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04


# install some necessary tools.
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update \
        && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        software-properties-common \
        g++ \
        sudo \
        ccache \
        cmake \
        libjpeg-dev \
        libpng-dev
RUN apt-get install -y \
        rsync \
        swig \
        curl \
        git \
        vim \
        wget \
        unzip \
        zsh \
        screen \
        tmux \
        openssh-server
RUN apt-get update && \
        apt-get install -y pciutils net-tools iputils-ping && \
        apt-get install -y htop && \
        rm -rf /var/lib/apt/lists/*

RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

# install good vim.
RUN curl http://j.mp/spf13-vim3 -L -o - | sh

# configure environments.
RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/*
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# configure user.
ENV SHELL=/bin/bash \
        NB_USER=lin \
        NB_UID=144057 \
        NB_GROUP=MLO-unit \
        NB_GID=11169
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
RUN echo 'lin ALL=(ALL) NOPASSWD: ALL' | sudo EDITOR='tee -a' visudo

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
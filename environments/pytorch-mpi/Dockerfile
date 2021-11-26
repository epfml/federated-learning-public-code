FROM user/base

USER $NB_USER
WORKDIR $HOME


# install openMPI
RUN mkdir $HOME/.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.3.tar.gz
RUN gunzip -c openmpi-4.0.3.tar.gz | tar xf - \
    && cd openmpi-4.0.3 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH

# install conda
ENV PYTHON_VERSION=3.8
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh ~/miniconda.sh -b -p $HOME/conda && \
    rm ~/miniconda.sh
RUN $HOME/conda/bin/conda update -n base conda
RUN $HOME/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION conda-build cython typing typing_extensions numpy pyyaml scipy ipython mkl mkl-include
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -c pytorch magma-cuda112
ENV PATH $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH
RUN $HOME/conda/bin/conda clean -ya

# install pytorch, torchvision, torchtext.
RUN git clone --recursive  https://github.com/pytorch/pytorch
RUN cd pytorch && \
    git checkout tags/v1.8.0 && \
    git submodule sync && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0;6.0;6.1;7.0;7.5;8.0;8.6" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which $HOME/conda/bin/conda))/../" \
    pip install -v .
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION  av -c conda-forge
RUN git clone https://github.com/pytorch/vision.git && cd vision && git checkout tags/v0.9.2-rc2 && $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py install
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install --upgrade git+https://github.com/pytorch/text

RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install spacy
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python -m spacy download en
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python -m spacy download de
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install nltk==3.4.5 transformers==2.2.2
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION scikit-learn
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install pytelegraf pymongo influxdb kubernetes jinja2

# install bit2byte.
RUN git clone https://github.com/tvogels/signSGD-with-Majority-Vote.git && \
    cd signSGD-with-Majority-Vote/main/bit2byte-extension/ && \
    $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py develop --user

# install other python related softwares.
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y protobuf
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y networkx
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c anaconda pandas
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c conda-forge tabulate
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install lmdb tensorboard_logger pyarrow msgpack msgpack_numpy mpi4py
RUN sudo apt-get update && sudo apt install -y libgl1-mesa-glx && sudo rm -rf /var/lib/apt/lists/*
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install POT opencv-python
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -c conda-forge python-blosc

# install the Hivemind.
RUN git clone https://github.com/learning-at-home/hivemind.git && \
    cd hivemind && $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install .

# clean.
RUN $HOME/conda/bin/conda clean -ya

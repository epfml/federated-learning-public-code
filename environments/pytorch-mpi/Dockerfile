FROM user/base

USER $NB_USER
WORKDIR $HOME

# install openMPI
RUN mkdir $HOME/.openmpi/
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz
RUN gunzip -c openmpi-4.1.4.tar.gz | tar xf - \
    && cd openmpi-4.1.4 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH

# install conda
ENV PYTHON_VERSION=3.8.12
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
RUN curl -fsSL -v -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p $HOME/conda && \
    rm ~/miniconda.sh
COPY pytorch-mpi/.condarc $HOME/conda/.condarc
RUN $HOME/conda/bin/conda update -n base conda
RUN $HOME/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y conda-build cython typing typing_extensions numpy pyyaml scipy ipython mkl mkl-include
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y astunparse ninja setuptools cmake cffi future six requests dataclasses
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -c pytorch magma-cuda112
ENV PATH $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH
RUN $HOME/conda/bin/conda clean -ya

# install pytorch, torchvision, torchtext.
RUN git clone --recursive https://github.com/pytorch/pytorch
RUN cd pytorch && \
    git checkout tags/v1.11.0 && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0 && \
    TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0;6.0;6.1;7.0;7.5;8.0;8.6" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which $HOME/conda/bin/conda))/../" \
    $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py install
RUN git clone https://github.com/pytorch/vision.git && cd vision && git checkout tags/v0.12.0 && $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py install
RUN git clone https://github.com/pytorch/text torchtext \
    && cd torchtext \
    && git checkout tags/v0.12.0 \
    && git submodule update --init --recursive \
    && $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py clean install

# install other packages.
RUN $HOME/conda/bin/conda install -y --name pytorch-py$PYTHON_VERSION -c conda-forge av ffmpeg tabulate python-blosc
RUN $HOME/conda/bin/conda install -y --name pytorch-py$PYTHON_VERSION scikit-learn protobuf networkx
RUN $HOME/conda/bin/conda install -y --name pytorch-py$PYTHON_VERSION -c anaconda pandas

RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install spacy
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python -m spacy download en_core_web_sm
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python -m spacy download de_core_news_sm

RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install nltk==3.4.5 transformers==2.2.2
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install pytelegraf pymongo influxdb kubernetes jinja2
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install lmdb tensorboard_logger pyarrow msgpack msgpack_numpy mpi4py

RUN sudo apt-get update && sudo apt install -y libgl1-mesa-glx && sudo rm -rf /var/lib/apt/lists/*
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install POT opencv-python

# clean.
RUN $HOME/conda/bin/conda clean -ya

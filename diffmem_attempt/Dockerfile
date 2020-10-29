FROM ubuntu:18.04

RUN apt-get update  \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
 libeigen3-dev \
 libfmt-dev \
 g++ \
 git \
 libsundials-dev \
 ca-certificates \
 cmake \
 cython3 \
 python3-dev \
 python3 \
 python3-numpy \
 python3-pandas \
 libtbb-dev \
 googletest \
 libcereal-dev \
 r-base \
 r-cran-rcpp \
 less \
 mc
# less and mc added to make it easier to explore the container internals

# install googletest
RUN cd /usr/src/googletest  \
 && cmake .  \
 && make install

WORKDIR /usr/src/

RUN git clone https://bitbucket.org/tomhaber/diffmem.git

RUN mkdir diffmem/build && cd diffmem/build &&\
    cmake \
    -DCYTHON_EXECUTABLE=/usr/bin/cython3 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DDIFFMEM_BUILD_TESTS=ON \
    -DCMAKE_CXX_FLAGS=-Wno-deprecated-declarations \
    ../ && \
    make -j 4

ENTRYPOINT /bin/bash
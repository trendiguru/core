#RUN echo deb http://archive.ubuntu.com/ubuntu precise universe multiverse >> /etc/apt/sources.list; \
sudo   apt-get update -qq && apt-get install -y --force-yes \
    curl \
    git \
    g++ \
    autoconf \
    automake \
    mercurial \
    libopencv-dev \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    libtiff4-dev \
    libpng-dev \
    libjpeg-dev \
    libjasper-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine-dev \
    libgstreamer0.10-dev \
    libgstreamer-plugins-base0.10-dev \
    libv4l-dev \
    libtbb-dev \
    libgtk2.0-dev \
    libfaac-dev \
    libmp3lame-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libtool \
    v4l-utils \
    python2.7 \
    python2.7-dev \
    python-numpy \
    default-jdk \
    ant \
    wget \
    unzip; \
    apt-get clean
YASM_VERSION=1.3.0
OPENCV_VERSION=3.0
cd /usr/local/src
git clone --depth 1 https://github.com/l-smash/l-smash
git clone --depth 1 git://git.videolan.org/x264.git
git clone https://bitbucket.org/multicoreware/x265
git clone --depth 1 git://source.ffmpeg.org/ffmpeg
git clone https://github.com/Itseez/opencv.git
git clone --depth 1 git://github.com/mstorsjo/fdk-aac.git
git clone --depth 1 https://chromium.googlesource.com/webm/libvpx
git clone --depth 1 git://git.opus-codec.org/opus.git
git clone --depth 1 https://github.com/mulx/aacgain.git
curl -Os http://www.tortall.net/projects/yasm/releases/yasm-${YASM_VERSION}.tar.gz
tar xzvf yasm-${YASM_VERSION}.tar.gz
# Build YASM # =================================
cd /usr/local/src/yasm-${YASM_VERSION}
./configure
make -j 4
make install
# ================================= # Build L-SMASH # =================================
cd /usr/local/src/l-smash
./configure
make -j 4
make install
# ================================= # Build libx264 # =================================
cd /usr/local/src/x264
./configure --enable-static
make -j 4
make install
# ================================= # Build libx265 # =================================
cd /usr/local/src/x265/build/linux
cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr ../../source
make -j 4
make install
# ================================= # Build libfdk-aac # =================================
cd /usr/local/src/fdk-aac
autoreconf -fiv
./configure --disable-shared
make -j 4
make install
# ================================= # Build libvpx # =================================
cd /usr/local/src/libvpx
./configure --disable-examples
make -j 4
make install
# ================================= # Build libopus # =================================
cd /usr/local/src/opus
./autogen.sh
./configure --disable-shared
make -j 4
make install
# ================================= # Build OpenCV 3.0.0 # =================================
cd /usr/local/src
apt-get update -qq 
#apt-get install -y --force-yes libopencv-dev
#git clone https://github.com/Itseez/opencv.git
wget https://github.com/Itseez/opencv/archive/3.0.0.zip
unzip 3.0.0.zip
#mkdir -p opencv/release
echo ls
cd opencv-3.0.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local     -D WITH_TBB=ON  -D BUILD_PYTHON_SUPPORT=ON ..

#cmake -D CMAKE_BUILD_TYPE=RELEASE \
 #         -D CMAKE_INSTALL_PREFIX=/usr/local \
  #        -D WITH_TBB=ON \
   #       -D BUILD_PYTHON_SUPPORT=ON \
    #      -D WITH_V4L=ON \
   #       ..
make -j4
make install
sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
ldconfig
# ================================= # Build ffmpeg. # =================================
apt-get update -qq && apt-get install -y --force-yes \
    libass-dev
cd /usr/local/src/ffmpeg
 ./configure --extra-libs="-ldl" \
            --enable-gpl \
            --enable-libass \
            --enable-libfdk-aac \
            --enable-libfontconfig \
            --enable-libfreetype \
            --enable-libfribidi \
            --enable-libmp3lame \
            --enable-libopus \
            --enable-libtheora \
            --enable-libvorbis \
            --enable-libvpx \
            --enable-libx264 \
            --enable-libx265 \
            --enable-nonfree
make -j 4
make install
# ================================= # Remove all tmpfile # =================================
cd /usr/local/
#rm -rf /usr/local/src

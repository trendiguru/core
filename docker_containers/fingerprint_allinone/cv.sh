mkdir -p /opt/OpenCV
cd /opt/OpenCV
wget -O OpenCV3.0.zip https://codeload.github.com/Itseez/opencv/zip/3.0.0
unzip OpenCV3.0.zip
wget -O opencv_contrib.zip https://codeload.github.com/Itseez/opencv_contrib/zip/3.0.0
unzip opencv_contrib.zip


#add the nonfree stuff here if necessary
mkdir -p /opt/OpenCV/opencv-3.0.0/build
cd $OPENCV_HOME/opencv-3.0.0/build
pwd
/opt/OpenCV/opencv-3.0.0/build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") -D PYTHON_EXECUTABLE=$(which python) -D BUILD_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_TESTS=OFF -D BUILD_opencv_java=OFF -D WITH_IPP=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON  -DWITH_QT=OFF ..

git clone https://bitbucket.org/tomhaber/diffmem.git
brew install eigen
brew install tbb
brew install sundials
brew install fmt
brew install cereal
brew install googletest
pip3 install cython
R -e 'require("Rcpp")'
cd diffmem
mkdir build # create build directory
cd build
cmake ../ # let cmake set up Makefiles
make
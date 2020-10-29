/*
 * Copyright (c) 2016, Hasselt University
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>
#include "TestCommon.hpp"

TEST(SliceTest, testSliceRows) {
	Matrixd x(5,5);
    x << 0.5377, -1.3077, -1.3499, -0.2050, 0.6715,
		1.8339, -0.4336,  3.0349, -0.1241, -1.2075,
	   -2.2588,  0.3426,  0.7254,  1.4897,  0.7172,
		0.8622,  3.5784, -0.0631,  1.4090,  1.6302,
		0.3188,  2.7694,  0.7147,  1.4172,  0.4889;

    Vecb mask(5); mask << true, false, true, true, false;

    Matrixd y;
    sliceRows(x, mask, y);

    EXPECT_EQ( y.rows(), 3 );
    EXPECT_EQ( y.cols(), 5 );

    Matrixd u(3,5);
    u << 0.5377, -1.3077, -1.3499, -0.2050, 0.6715,
	   -2.2588,  0.3426,  0.7254,  1.4897,  0.7172,
		0.8622,  3.5784, -0.0631,  1.4090,  1.6302;

    AssertEqual(u, y);
}

TEST(SliceTest, testSliceCols) {
	Matrixd x(5,5);
    x << 0.5377, -1.3077, -1.3499, -0.2050, 0.6715,
		1.8339, -0.4336,  3.0349, -0.1241, -1.2075,
	   -2.2588,  0.3426,  0.7254,  1.4897,  0.7172,
		0.8622,  3.5784, -0.0631,  1.4090,  1.6302,
		0.3188,  2.7694,  0.7147,  1.4172,  0.4889;

    Vecb mask(5); mask << true, false, true, true, false;

    Matrixd y;
    sliceCols(x, mask, y);

    EXPECT_EQ( y.rows(), 5 );
    EXPECT_EQ( y.cols(), 3 );

    Matrixd u(5,3);
    u << 0.5377,-1.3499, -0.2050,
		1.8339,  3.0349, -0.1241,
	   -2.2588,  0.7254,  1.4897,
		0.8622, -0.0631,  1.4090,
		0.3188,  0.7147,  1.4172;

    AssertEqual(u, y);
}

TEST(SliceTest, testSliceRowsAndCols) {
	Matrixd x(5,5);
    x << 0.5377, -1.3077, -1.3499, -0.2050, 0.6715,
		1.8339, -0.4336,  3.0349, -0.1241, -1.2075,
	   -2.2588,  0.3426,  0.7254,  1.4897,  0.7172,
		0.8622,  3.5784, -0.0631,  1.4090,  1.6302,
		0.3188,  2.7694,  0.7147,  1.4172,  0.4889;

    Vecb mask(5); mask << true, false, true, true, false;

    Matrixd y;
    slice(x, mask, y);

    EXPECT_EQ( y.rows(), 3 );
    EXPECT_EQ( y.cols(), 3 );

    Matrixd u(3,3);
    u << 0.5377,-1.3499, -0.2050,
	   -2.2588,  0.7254,  1.4897,
		0.8622, -0.0631,  1.4090;

    AssertEqual(u, y);
}

TEST(SliceTest, testSliceVector) {
	Matrixd x(5,1);
    x << 0.5377, -1.3077, -1.3499, -0.2050, 0.6715;

    Vecb mask(5); mask << true, false, true, true, false;

    Matrixd y;
    slice(x, mask, y);

    EXPECT_EQ( y.rows(), 3 );
    EXPECT_EQ( y.cols(), 1 );

    Matrixd u(3,1);
    u << 0.5377, -1.3499, -0.2050;

    AssertEqual(u, y);
}

TEST(SliceTest, testSliceIntoRows) {
    Matrixd x(3,5);
    x << 0.5377, -1.3077, -1.3499, -0.2050, 0.6715,
	   -2.2588,  0.3426,  0.7254,  1.4897,  0.7172,
		0.8622,  3.5784, -0.0631,  1.4090,  1.6302;

    Vecb mask(5); mask << true, false, true, true, false;

	Matrixd y(5,5);
	y.setZero();

    sliceIntoRows(x, mask, y);

    Matrixd u(5,5);
    u << 0.5377, -1.3077, -1.3499, -0.2050, 0.6715,
      	0.0,     0.0,     0.0,     0.0,     0.0,
	   -2.2588,  0.3426,  0.7254,  1.4897,  0.7172,
		0.8622,  3.5784, -0.0631,  1.4090,  1.6302,
      	0.0,     0.0,     0.0,     0.0,     0.0;

    AssertEqual(u, y);
}

TEST(SliceTest, testSliceIntoCols) {
	Matrixd x(5,3);
    x << 0.5377,-1.3499, -0.2050,
		1.8339,  3.0349, -0.1241,
	   -2.2588,  0.7254,  1.4897,
		0.8622, -0.0631,  1.4090,
		0.3188,  0.7147,  1.4172;

    Vecb mask(5); mask << true, false, true, true, false;

    Matrixd y(5,5);
	y.setZero();

	sliceIntoCols(x, mask, y);

    EXPECT_EQ( y.rows(), 5 );
    EXPECT_EQ( y.cols(), 5 );

    Matrixd u(5,5);
    u << 0.5377, 0.0,  -1.3499, -0.2050, 0.0,
    	1.8339,  0.0,   3.0349, -0.1241, 0.0,
	   -2.2588,  0.0,   0.7254,  1.4897, 0.0,
		0.8622,  0.0,  -0.0631,  1.4090, 0.0,
		0.3188,  0.0,   0.7147,  1.4172, 0.0;

    AssertEqual(u, y);
}

TEST(SliceTest, testSliceIntoRowsAndCols) {
	Matrixd x(3,3);
    x << 0.5377,-1.3499, -0.2050,
	   -2.2588,  0.7254,  1.4897,
		0.8622, -0.0631,  1.4090;

    Vecb mask(5); mask << true, false, true, true, false;

    Matrixd y(5,5);
	y.setZero();

    sliceInto(x, mask, y);

    Matrixd u(5,5);
    u << 0.5377, 0.0,  -1.3499, -0.2050, 0.0,
    	0.0,     0.0,   0.0,     0.0,    0.0,
	   -2.2588,  0.0,   0.7254,  1.4897, 0.0,
		0.8622,  0.0,  -0.0631,  1.4090, 0.0,
		0.0,     0.0,   0.0,     0.0,    0.0;

    AssertEqual(u, y);
}

TEST(SliceTest, testSliceIntoVector) {
    Matrixd x(3,1);
    x << 0.5377, -1.3499, -0.2050;

    Vecb mask(5); mask << true, false, true, true, false;

    Matrixd y(5,1);
	y.setZero();

    sliceInto(x, mask, y);

    Matrixd u(5,1);
    u << 0.5377, 0.0, -1.3499, -0.2050, 0.0;

    AssertEqual(u, y);
}

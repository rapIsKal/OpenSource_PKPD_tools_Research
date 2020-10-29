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
#include "FiniteDifference.hpp"
#include "TestCommon.hpp"

TEST(FiniteDifferenceTest, testGradient) {
	auto rosenbrock = [](const VecNd<2> &x) -> double {
		const double t1 = (1 - x[0]);
		const double t2 = (x[1] - x[0] * x[0]);
		return t1 * t1 + 100 * t2 * t2;
	};

	VecNd<2> x(0.0, 0.0);
	VecNd<2> grad;
	FiniteDifference::computeGradient(rosenbrock, x, grad);
	EXPECT_NEAR( -2.0 * (1.0 - x[0]) - 400.0 * (x[1] - x[0]*x[0]) * x[0], grad[0], 1e-4 );
	EXPECT_NEAR( 200.0 * (x[1] - x[0]*x[0]), grad[1], 1e-4 );
}

void testHessian(int order) {
	auto rosenbrock = [](const VecNd<2> &x) -> double {
		const double t1 = (1 - x[0]);
		const double t2 = (x[1] - x[0] * x[0]);
		return t1 * t1 + 100 * t2 * t2;
	};

	VecNd<2> x(0.0, 0.0);
	MatrixMNd<2,2> H;
	double val = FiniteDifference::computeHessian(rosenbrock, x, H, order);
	EXPECT_NEAR(val, rosenbrock(x), 1e-14);
	EXPECT_NEAR( 2.0 - 400.0 * x[1] + 1200.0 * x[0]*x[0], H(0,0), 1e-3 );
	EXPECT_NEAR( -400.0 * x[0], H(0,1), 1e-3 );
	EXPECT_NEAR( -400.0 * x[0], H(1,0), 1e-3 );
	EXPECT_NEAR( 200.0, H(1,1), 1e-3 );
}

void testHessianBig(int order) {
	MatrixMNd<5,5> A;
	A << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	VecNd<5> x;
	x << -0.6693245,  0.2262109,  1.3699085 ,-1.5203059 ,-0.1958099;

	MatrixMNd<5,5> H;
	FiniteDifference::computeHessian(
		[&A](const VecNd<5> & x) -> double {
			return 0.5 * x.transpose() * A * x;
		}, x, H, order, 1e-4, 1e-6);

	AssertEqual(A, H, 1e-3);
}

TEST(FiniteDifferenceTest, testHessian_1) {
	testHessian(1);
}

TEST(FiniteDifferenceTest, testHessianBig_1) {
	testHessianBig(1);
}

TEST(FiniteDifferenceTest, testHessian_2) {
	testHessian(2);
}

TEST(FiniteDifferenceTest, testHessianBig_2) {
	testHessianBig(2);
}

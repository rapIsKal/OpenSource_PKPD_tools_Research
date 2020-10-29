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
#include "stats/wishart.hpp"
#include "Random.hpp"
#include "TestCommon.hpp"

TEST(WishartTest, testDiag) {
	Matrixd X(5,5);
	X << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	DiagonalMatrixd S(5);
	S.diagonal() << 1.0, 1.0, 1.0, 1.0, 1.0;
	EXPECT_NEAR( -23.61085, stats::Wishart::logpdf<false>(X, 5.0, S), 1e-2 );

	S.diagonal() << .8, 0.2, 1.0, 0.5, 1.0;
	EXPECT_NEAR( -26.59229, stats::Wishart::logpdf<false>(X, 5.0, S), 1e-2 );

	EXPECT_NEAR( -29.6475, stats::Wishart::logpdf<false>(X, 10.0, S), 1e-2 );

	EXPECT_NEAR( -29.6475, stats::Wishart::logpdf<false>(Cholesky<Matrixd>(X), 10.0, S), 1e-2 );
}

TEST(WishartTest, testNonDiag) {
	Matrixd X(5,5);
	X << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	Matrixd S(5,5);
	S << 0.9633, 0.1120, 0.0989,   -0.3160,    0.6690,
		    0.1120,    0.3417,    0.1158,    0.1509,   -0.1765,
		    0.0989,    0.1158,    0.4398,    0.1384,   -0.2221,
		   -0.3160,    0.1509,    0.1384,    0.9759,   -1.1745,
		    0.6690,   -0.1765,   -0.2221,   -1.1745,    2.0059;

	EXPECT_NEAR( -36.69965, stats::Wishart::logpdf<false>(X, 5.0, S), 1e-4 );
	EXPECT_NEAR( -36.69965, stats::Wishart::logpdf<false>(X, 5.0, Cholesky<Matrixd>(S)), 1e-4 );
	EXPECT_NEAR( -36.69965, stats::Wishart::logpdf<false>(Cholesky<Matrixd>(X), 5.0, Cholesky<Matrixd>(S)), 1e-4 );
}

TEST(WishartTest, testRandom) {
	Random rng;

	Matrixd S(5,5);
	S << 0.9633, 0.1120, 0.0989,   -0.3160,    0.6690,
		    0.1120,    0.3417,    0.1158,    0.1509,   -0.1765,
		    0.0989,    0.1158,    0.4398,    0.1384,   -0.2221,
		   -0.3160,    0.1509,    0.1384,    0.9759,   -1.1745,
		    0.6690,   -0.1765,   -0.2221,   -1.1745,    2.0059;

	Matrixd omega;
	stats::Wishart::sample(rng, 5.0, S, omega);

	Cholesky<Matrixd> chol_omega( omega );
	EXPECT_TRUE( chol_omega.info() == Eigen::Success );

	// TODO actual wishart test
}

TEST(WishartTest, testRandomDiag) {
	Random rng;

	DiagonalMatrixd S(5);
	S.diagonal() << 1.0, 1.0, 1.0, 1.0, 1.0;
	//S.diagonal() << .8, 0.2, 1.0, 0.5, 1.0;

	Matrixd omega;
	stats::Wishart::sample(rng, 5.0, S, omega);

	Cholesky<Matrixd> chol_omega( omega );
	EXPECT_TRUE( chol_omega.info() == Eigen::Success );

	// TODO actual wishart test
}

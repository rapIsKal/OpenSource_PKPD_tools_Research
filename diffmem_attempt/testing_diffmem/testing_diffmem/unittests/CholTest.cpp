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
#include "Cholesky.hpp"

TEST(CholTest, testInverse) {
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	Matrixd omega_inv(5,5);
	omega_inv << 0.9633, 0.1120, 0.0989,   -0.3160,    0.6690,
			0.1120,    0.3417,    0.1158,    0.1509,   -0.1765,
			0.0989,    0.1158,    0.4398,    0.1384,   -0.2221,
		   -0.3160,    0.1509,    0.1384,    0.9759,   -1.1745,
			0.6690,   -0.1765,   -0.2221,   -1.1745,    2.0059;

	Matrixd inv;
	chol_inv(omega, inv);
	AssertEqual(omega_inv, inv, 1e-3);
}

TEST(CholTest, testProduct) {
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	Cholesky<Matrixd> chol_omega(omega);

	Vecd x(5);
	x << 1.5326,-0.7697, 0.3714, -0.2256, 1.1174;

	Vecd y = chol_omega.solve(x);

	Vecd res(5);
	res << 2.2457, -0.2796, -0.0535, -2.0816, 3.5851;

	AssertEqual(res, y, 1e-3);
}

TEST(CholTest, testDeterminant) {
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	Cholesky<Matrixd> chol_omega(omega);
	EXPECT_NEAR(23.347369396646549, determinant(chol_omega), 1e-8);
}

TEST(CholTest, testLogDeterminant) {
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	Cholesky<Matrixd> chol_omega(omega);
	EXPECT_NEAR(3.150484318095804, logDeterminant(chol_omega), 1e-4);
}

TEST(CholTest, testSemi) {
	Matrixd omega(3,3);
	omega <<  1.4784946278818258,  1.1736377759164127,  1.0203098082187532,
			1.1736377759164127,  1.4982567148520356,   0.8888263003281415,
			1.0203098082187532,  0.8888263003281415,  0.7151024413333396;

	Matrixd omega_L(3,3);
	omega_L <<  1.21593, 0.965215, 0.839116,
			 0.965215,  0.752739,   0.104815,
			 0.839116,  0.104815,  2.10734e-8;

	Cholesky<Matrixd> chol_omega(omega);
	AssertEqual(chol_omega.matrixL(), omega_L, 1e-3);
}

TEST(CholTest, testExtractCoefficientsLower) {
	Matrixd omega_L(3,3);
	omega_L <<  0.1, 0.0, 0.0,
			 0.2,  0.4,   0.0,
			 0.3,  0.5,  0.6;

	Cholesky<Matrixd> chol_omega;
	chol_omega.set(omega_L);

	Vecd c;
	chol_omega.extractCoefficients(c);

	EXPECT_EQ(Cholesky<Matrixd>::numberOfCoefficients(3), 6);
	EXPECT_EQ(c.size(), 6);
	EXPECT_NEAR(c[0], .1, 1e-15);
	EXPECT_NEAR(c[1], .2, 1e-15);
	EXPECT_NEAR(c[2], .3, 1e-15);
	EXPECT_NEAR(c[3], .4, 1e-15);
	EXPECT_NEAR(c[4], .5, 1e-15);
	EXPECT_NEAR(c[5], .6, 1e-15);
}

TEST(CholTest, testFromCoefficientsLower) {
	Matrixd omega_L(3,3);
	omega_L <<  0.1, 0.0, 0.0,
			 0.2,  0.4,   0.0,
			 0.3,  0.5,  0.6;

	Vecd c(6);
	c << .1, .2, .3, .4, .5, .6;

	Cholesky<Matrixd> chol_omega;
	chol_omega.fromCoefficients(c);

	EXPECT_EQ(Cholesky<Matrixd>::coefficientsToDim(6), 3);
	EXPECT_EQ(chol_omega.rows(), 3);
	AssertEqual(chol_omega.matrixL(), omega_L, 1e-15);
}

TEST(CholTest, testExtractCoefficientsUpper) {
	Matrixd omega_U(3,3);
	omega_U <<  0.1, 0.2, 0.4,
			 0.0,  0.3,   0.5,
			 0.0,  0.0,  0.6;

	Cholesky<Matrixd, Eigen::Upper> chol_omega;
	chol_omega.set(omega_U);

	Vecd c;
	chol_omega.extractCoefficients(c);

	EXPECT_EQ(Cholesky<Matrixd>::numberOfCoefficients(3), 6);
	EXPECT_EQ(c.size(), 6);
	EXPECT_NEAR(c[0], .1, 1e-15);
	EXPECT_NEAR(c[1], .2, 1e-15);
	EXPECT_NEAR(c[2], .3, 1e-15);
	EXPECT_NEAR(c[3], .4, 1e-15);
	EXPECT_NEAR(c[4], .5, 1e-15);
	EXPECT_NEAR(c[5], .6, 1e-15);
}

TEST(CholTest, testFromCoefficientsUpper) {
	Matrixd omega_U(3,3);
	omega_U <<  0.1, 0.2, 0.4,
			 0.0,  0.3,   0.5,
			 0.0,  0.0,  0.6;

	Vecd c(6);
	c << .1, .2, .3, .4, .5, .6;

	Cholesky<Matrixd, Eigen::Upper> chol_omega;
	chol_omega.fromCoefficients(c);

	EXPECT_EQ(Cholesky<Matrixd>::coefficientsToDim(6), 3);
	EXPECT_EQ(chol_omega.rows(), 3);
	AssertEqual(chol_omega.matrixU(), omega_U, 1e-15);
}
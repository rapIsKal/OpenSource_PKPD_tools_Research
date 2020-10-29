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
#include "stats/mvnormal.hpp"

template <typename XT, typename MT, typename OT>
void testMVNDistribution(const XT & x, const MT & mu, const OT & omega, double expected) {
	const double val = stats::MVNormal::logpdf<false>(x, mu, omega);
	EXPECT_SIMILAR( expected, val, 1e-6 );

	const double valProp = stats::MVNormal::logpdf<true>(x, mu, omega);

	Vecd xx = x.array() + 1e-4;
	const double valEps = stats::MVNormal::logpdf<false>(xx, mu, omega);
	const double valPropEps = stats::MVNormal::logpdf<true>(xx, mu, omega);
	EXPECT_SIMILAR( valEps - val, valPropEps - valProp, 1e-6 );
}

template <typename XT, typename OT>
void testMVNDistribution(const XT & x, const OT & omega, double expected) {
	const double val = stats::MVNormal::logpdf<false>(x, omega);
	EXPECT_SIMILAR( expected, val, 1e-6 );

	const double valProp = stats::MVNormal::logpdf<true>(x, omega);

	Vecd xx = x.array() + 1e-4;
	const double valEps = stats::MVNormal::logpdf<false>(xx, omega);
	const double valPropEps = stats::MVNormal::logpdf<true>(xx, omega);
	EXPECT_SIMILAR( valEps - val, valPropEps - valProp, 1e-6 );
}

TEST(Stats, testMVN) {
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;

	Vecd x(5);
	x << -0.6693245,  0.2262109,  1.3699085 ,-1.5203059 ,-0.1958099;

	Vecd mu = Vecd::Zero(5);

	testMVNDistribution(x, mu, omega, -7.045233);
	testMVNDistribution(x, omega, -7.045233);
}

TEST(Stats, testMVNVec) {
	DiagonalMatrixd omega(5);
	omega.diagonal() << .8, 0.2, 1.0, 0.5, 1.0;

	Vecd x(5);
	x << -0.6693245,  0.2262109,  1.3699085 ,-1.5203059 ,-0.1958099;

	Vecd mu = Vecd::Zero(5);

	testMVNDistribution(x, mu, omega, -7.008579);
	testMVNDistribution(x, omega, -7.008579);
}

TEST(Stats, testMVNChol) {
	Matrixd omega(5,5);
	omega << 1.6613,   -0.6686,   -0.5310,   -0.3107,   -0.8536,
		   -0.6686,    3.6404,   -0.5567,   -0.4084,    0.2426,
		   -0.5310,   -0.5567,    2.7555,    0.1509,    0.5215,
		   -0.3107,   -0.4084,    0.1509,    3.6146,    2.2008,
		   -0.8536,    0.2426,    0.5215,    2.2008,    2.1509;
	Cholesky<Matrixd> chol_omega( omega );

	Vecd x(5);
	x << -0.6693245,  0.2262109,  1.3699085 ,-1.5203059 ,-0.1958099;

	Vecd mu = Vecd::Zero(5);

	testMVNDistribution(x, mu, omega, -7.045233);
	testMVNDistribution(x, omega, -7.045233);
}

template <typename XT, typename MT, typename OT>
void testMVNDistributionInv(const XT & x, const MT & mu, const OT & omega, double expected) {
	const double val = stats::MVNormal::logpdf_inv<false>(x, mu, omega);
	EXPECT_SIMILAR( expected, val, 1e-6 );

	const double valProp = stats::MVNormal::logpdf_inv<true>(x, mu, omega);

	Vecd xx = x.array() + 1e-4;
	const double valEps = stats::MVNormal::logpdf_inv<false>(xx, mu, omega);
	const double valPropEps = stats::MVNormal::logpdf_inv<true>(xx, mu, omega);
	EXPECT_SIMILAR( valEps - val, valPropEps - valProp, 1e-6 );
}

template <typename XT, typename OT>
void testMVNDistributionInv(const XT & x, const OT & omega, double expected) {
	const double val = stats::MVNormal::logpdf_inv<false>(x, omega);
	EXPECT_SIMILAR( expected, val, 1e-6 );

	const double valProp = stats::MVNormal::logpdf_inv<true>(x, omega);

	Vecd xx = x.array() + 1e-4;
	const double valEps = stats::MVNormal::logpdf_inv<false>(xx, omega);
	const double valPropEps = stats::MVNormal::logpdf_inv<true>(xx, omega);
	EXPECT_SIMILAR( valEps - val, valPropEps - valProp, 1e-6 );
}
TEST(Stats, testMVNCholInv) {
	Matrixd inv_omega(5,5);
	inv_omega <<  0.963287,    0.112019,   0.0989563,  -0.315995,   0.668985,
		0.112019,   0.341662,   0.11576,     0.150881,  -0.176529,
		0.0989563,   0.11576,    0.439813,    0.13843,   -0.222062,
		-0.315995,    0.150881,   0.13843,     0.975871,  -1.1745,
		0.668985,   -0.176529,  -0.222062,   -1.1745,     2.00591;
	Cholesky<Matrixd> chol_inv_omega( inv_omega );

	Vecd x(5);
	x << -0.6693245,  0.2262109,  1.3699085 ,-1.5203059 ,-0.1958099;

	Vecd mu = Vecd::Zero(5);

	testMVNDistributionInv(x, mu, chol_inv_omega, -7.0452362439733847);
	testMVNDistributionInv(x, chol_inv_omega, -7.0452362439733847);
}


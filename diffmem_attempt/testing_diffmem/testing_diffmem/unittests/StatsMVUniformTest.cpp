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
#include "stats/mvuniform.hpp"

template <typename XT, typename LT, typename UT>
void testMVUDistribution(const XT & x, const LT & lb, const UT & ub, double expected) {
	const double val = stats::MVUniform::logpdf<false>(x, lb, ub);
	EXPECT_SIMILAR( expected, val, 1e-6 );

	const double valProp = stats::MVUniform::logpdf<true>(x, lb, ub);

	Vecd xx = x.array() + 1e-4;
	const double valEps = stats::MVUniform::logpdf<false>(xx, lb, ub);
	const double valPropEps = stats::MVUniform::logpdf<true>(xx, lb, ub);
	EXPECT_SIMILAR( valEps - val, valPropEps - valProp, 1e-6 );
}

TEST(Stats, testMVUniform) {
	Vecd x(5);
	x << 0.257943, 0.0328989, -0.433012, -0.00997072, 0.302913;

	Vecd lb(5);
	lb << 0.221811, -0.386355, -0.781731, -0.324012, -0.206326;
	Vecd ub(5);
	ub <<  0.370524, 0.165674, 0.768498, 0.883178, 0.885736;

	testMVUDistribution(x, lb, ub, 1.7851260438188479);
}

TEST(Stats, testMVUniformRng) {
	Vecd x(5);

	Vecd lb(5);
	lb << 0.221811, -0.386355, -0.781731, -0.324012, -0.206326;
	Vecd ub(5);
	ub <<  0.370524, 0.165674, 0.768498, 0.883178, 0.885736;

	Random rng;
	for(int i = 0; i < 10; ++i) {
		stats::MVUniform::sample(rng, lb, ub, x);
		EXPECT_TRUE(math::isFinite(stats::MVUniform::logpdf<false>(x, lb, ub)));
	}
}
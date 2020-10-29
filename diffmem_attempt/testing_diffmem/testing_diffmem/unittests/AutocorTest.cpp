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
#include "Statistics.hpp"

TEST(AutocorTest, testAutocorr) {
	Vecd x(5);
	x << 1, 2, 3, 4, 5;

	Vecd ac;
	Statistics::autocorrelation(x, ac);

	EXPECT_NEAR(1.0, ac[0], 1e-8);
	EXPECT_NEAR(0.4, ac[1], 1e-8);
	EXPECT_NEAR(-0.1, ac[2], 1e-8);
	EXPECT_NEAR(-0.4, ac[3], 1e-8);
	EXPECT_NEAR(-0.4, ac[4], 1e-8);
}

TEST(AutocorTest, testAutocov) {
	Vecd x(5);
	x << 1, 2, 3, 4, 5;

	Vecd ac;
	Statistics::autocovariance(x, ac);

	EXPECT_NEAR(2.0, ac[0], 1e-8);
	EXPECT_NEAR(0.8, ac[1], 1e-8);
	EXPECT_NEAR(-0.2, ac[2], 1e-8);
	EXPECT_NEAR(-0.8, ac[3], 1e-8);
	EXPECT_NEAR(-0.8, ac[4], 1e-8);
}


TEST(AutocorTest, testESS) {
	auto x = make_vector<double>({101.95157, 99.30803, 97.03714, 98.08499, 94.20236, 98.28347, 97.99453, 97.36125, 105.98848, 103.08103});

	double ess = Statistics::effectiveSampleSize(x);
	EXPECT_NEAR(ess, 7.365125984694775, 1e-5);
}

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
#include "math/number.hpp"
#include "math/digamma.hpp"

// digamma returns `NaN` if provided `0`
TEST(Math, testDigamma0) {
	double val = math::digamma(0.0);
	EXPECT_TRUE( math::isNaN(val) );
}

// test the digamma function for `x` such that remainder > 0.5
TEST(Math, testDigammaRemainder) {
	double val = math::digamma(-3.8);
	EXPECT_NEAR(-2.863183589156929, val, 1e-14);
}

TEST(Math, testDigammaSmall) {
	double val;

	val = math::digamma(0.000001);
	EXPECT_NEAR(-1000001, val, 1);

	val = math::digamma(0.0000005);
	EXPECT_NEAR(-2000001, val, 1);
}

TEST(Math, testDigamma) {
	double params[] = {
		-6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1,
		1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5,
		12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5,
		21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25, 1000, 10000,
		273.1, 1420.12, 7384.62, 38400.02, 199680.1, 1038336.52, 5399349.9, 28076619.48, 145998421.3,
		759191790.76, 3947797311.95, 20528546022.14
	};

	double values[] = {
		1.946757484246087, math::NaN(), 1.792911330399933, math::NaN(), 1.611093148581751, math::NaN(),
		1.388870926359529, math::NaN(), 1.103156640645243, math::NaN(), 0.7031566406452432, math::NaN(),
		0.03648997397857652, math::NaN(), -1.963510026021423, -0.5772156649015329, 0.03648997397857652,
		0.4227843350984671, 0.7031566406452432, 0.9227843350984671, 1.103156640645243, 1.256117668431800, 1.388870926359529,
		1.506117668431800, 1.611093148581751, 1.706117668431800, 1.792911330399933, 1.872784335098467, 1.946757484246087,
		2.015641477955610, 2.080090817579420, 2.140641477955610, 2.197737876402950, 2.251752589066721, 2.303001034297686,
		2.351752589066721, 2.398239129535782, 2.442661679975812, 2.485195651274912, 2.525995013309145, 2.565195651274912,
		2.602918090232222, 2.639269725348986, 2.674346661660794, 2.708235242590365, 2.741013328327460, 2.772751371622623,
		2.803513328327460, 2.833357432228684, 2.862336857739225, 2.890500289371541, 2.917892413294781, 2.944554343425595,
		2.970523992242149, 2.995836394707647, 3.020523992242149, 3.044616882512525, 3.068143039861197, 3.091128510419501,
		3.113597585315742, 3.135572954863946, 3.157075846185307, 3.178126146353308, 3.198742512851974, 6.907255195648812,
		9.210290371142849,
		5.608006079969503, 7.258144529823678, 8.907087028169087, 10.55580023852595, 12.20446936095711,
		13.853130009012208, 15.501789022794545, 17.15044772317726, 18.799106363175756, 20.447764991529237,
		22.096423617648053, 23.745082243337734,
	};

	const int N = sizeof(values) / sizeof(values[0]);
	for(int i = 0; i < N; ++i) {
		const double val = math::digamma(params[i]);
		EXPECT_SIMILAR_OR_NAN(values[i], val, epsilon(val, 1e-15, 1e-15));
	}
}

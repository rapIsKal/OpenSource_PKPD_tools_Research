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
#include "math/interp.hpp"
#include <type_traits>

TEST(Math, testInterpSame) {
	auto xs = make_vector<double>({1, 2, 3, 4, 5});
	auto interp = math::interp(xs, xs, 2.0);
	static_assert(std::is_same<decltype(interp), double>::value, "type should be double");
	EXPECT_NEAR(interp, 2.0, 1e-15);

	interp = math::interp(xs, xs, 2.5);
	EXPECT_NEAR(interp, 2.5, 1e-15);
}

TEST(Math, testInterp) {
	auto xs = make_vector<double>({1, 3, 4, 5, 6});
	auto ys = make_vector<double>({5, 4, 2, 8, 7});

	struct {
		double x;
		double y;
	} vals[] = {
		{0.0, 5.0},
		{1.0, 5.0},
		{6.0, 7.0},
		{7.0, 7.0},
		{3.0, 4.0},
		{3.5, 3.0},
		{4.5, 5.0},
		{3.2, 3.6},
		{5.23, 7.77},
		{5.9, 7.1}
	};

	for(int i = 0; i < 10; ++i) {
		auto interp = math::interp(xs, ys, vals[i].x);
		EXPECT_NEAR(interp, vals[i].y, 1e-15);
	}
}

TEST(Math, testInterpConstant) {
	auto xs = make_vector<double>({1, 3, 4, 5, 6});
	auto ys = make_vector<double>({5, 4, 2, 8, 7});

	struct {
		double x;
		double y;
	} vals[] = {
		{0.0, 5.0},
		{1.0, 5.0},
		{6.0, 7.0},
		{7.0, 7.0},
		{3.0, 4.0},
		{3.5, 4.0},
		{4.5, 2.0},
		{3.2, 4.0},
		{5.23, 8.0},
		{5.9, 8.0}
	};

	for(int i = 0; i < 10; ++i) {
		auto interp = math::interp(xs, ys, vals[i].x, math::constant_interp{});
		EXPECT_NEAR(interp, vals[i].y, 1e-15);
	}
}

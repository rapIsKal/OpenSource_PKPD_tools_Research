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
#include "MaskedFunction.hpp"
#include "MaskedDistribution.hpp"
#include "RosenbrockDistribution.hpp"

namespace {
	class Rosenbrock : public Function {
		public:
			double operator ()(ConstRefVecd &x) const override {
				const double t1 = (1 - x[0]);
				const double t2 = (x[1] - x[0] * x[0]);
				return t1 * t1 + 100 * t2 * t2;
			}

			double operator ()(ConstRefVecd &x, RefVecd grad) const override {
				const double t1 = (1 - x[0]);
				const double t2 = (x[1] - x[0] * x[0]);
				grad[0] = -2.0 * t1 - 400.0 * t2 * x[0];
				grad[1] = 200.0 * t2;
				return t1 * t1 + 100 * t2 * t2;
			}
	};
}

TEST(Masking, testFunction) {
	Rosenbrock f;
	MaskedFunction<2> mf(f, make_vector<double>({.5,0.0}), make_vector<bool>({false, true}));
	EXPECT_EQ(f(make_vector({0.5, 0.0})), mf(make_vector({0.0})));
}

TEST(Masking, testDistribution) {
	auto rosenbrock = std::make_shared<RosenbrockDistribution<2>>();
	auto masked = mask_distribution(rosenbrock, make_vector<double>({.5,0.0}), make_vector<bool>({false, true}));
	EXPECT_EQ(rosenbrock->logpdf(make_vector({0.5, 0.0})), masked->logpdf(make_vector({0.0})));
}

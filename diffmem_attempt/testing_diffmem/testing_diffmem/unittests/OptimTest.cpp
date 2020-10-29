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
#include "BFGS.hpp"
#include "NelderMead.hpp"
#include "PatternSearch.hpp"

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

	class PsExample : public Function {
		public:
			double operator()(ConstRefVecd &x) const override {
				if( x[0] < -5 )
					return (x[0]+5)*(x[0]+5) + std::abs(x[1]);
				else if( x[0] < -3 )
					return -2*std::sin(x[0]) + std::abs(x[1]);
				else if( x[0] < 0 )
					return 0.5*x[0] + 2 + std::abs(x[1]);
				else
					return 0.3*std::sqrt(x[0]) + 5./2. +std::abs(x[1]);
			}
	};

}

TEST(OptimTest, testRosenbrockBFGS) {
	Rosenbrock f;
	Vecd initial_x(2);
	initial_x << 0.0, 0.0;
	BFGS opt(f, initial_x);
	opt.optimize();

	EXPECT_TRUE( opt.isConverged() );
	EXPECT_NEAR( 1.0, opt.optimum()[0], 1e-8 );
	EXPECT_NEAR( 1.0, opt.optimum()[1], 1e-8 );
	EXPECT_NEAR( 0.0, opt.value(), 1e-8 );
}

TEST(OptimTest, testRosenbrockNelderMead) {
	Rosenbrock f;
	Vecd initial_x(2);
	initial_x << 0.0, 0.0;
	NelderMead opt(f, initial_x);
	opt.optimize();

	EXPECT_TRUE( opt.isConverged() );
	EXPECT_NEAR( 1.0, opt.optimum()[0], 1e-8 );
	EXPECT_NEAR( 1.0, opt.optimum()[1], 1e-8 );
	EXPECT_NEAR( 0.0, opt.value(), 1e-8 );
}

TEST(OptimTest, testRosenbrockPattern) {
	PsExample f;
	Vecd initial_x(2);
	initial_x << 2.1, 1.7;

	PatternSearch opt(f, initial_x);
	opt.optimize();

	EXPECT_TRUE( opt.isConverged() );
	EXPECT_NEAR( -4.71238, opt.optimum()[0], 1e-5 );
	EXPECT_NEAR( 0.0, opt.optimum()[1], 1e-6 );
	EXPECT_NEAR( -2.0, opt.value(), 1e-6 );
}

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
#include "ParticleSwarmOptimization.hpp"

template <int dim>
class Rosenbrock : public Function {
	public:
		double operator()(ConstRefVecd & x) const override {
			auto s1 = x.segment(0, dim-1);
			auto s2 = x.segment(1, dim-1);
			auto e1 = (s2 - s1.cwiseProduct(s1));
			auto e2 = (1.0 - s1.array()).matrix();
			return (100 * e1.cwiseProduct(e1) + e2.cwiseProduct(e2)).sum();
		}
};

TEST(ParticleSwarmOptimizationTest, testRosenbrock) {
	constexpr int dim = 6;
	Rosenbrock<dim> f;
	Vecd lb(dim), ub(dim);
	for(int i = 0; i < dim; ++i) {
		lb[i] = -10.0;
		ub[i] = 10.0;
	}

	ParticleSwarmOptimization::Options opts;
	opts.numEpochs = 1000;
	opts.numParticles = 100;

	// Added because of paper named "Good Parameters for Particle Swarm Optimization" (Pedersen, 2010)
	opts.c1 = 3.1913;
	opts.c2 = 0.5287;
	opts.w_start = -0.1832;

	ParticleSwarmOptimization opt(f, opts);
	opt.optimize(lb, ub);

	EXPECT_TRUE( opt.value() < 1.0 );
}

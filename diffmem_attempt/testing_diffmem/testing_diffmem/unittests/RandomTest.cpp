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
#include "Random.hpp"

struct basic_dist {
	typedef uint32_t result_type;

	template <typename R>
	result_type operator() (R &r) const {
		return r();
	}
};

uint32_t generate(Random &rng, size_t skip=0) {
	basic_dist f;

	uint32_t x = 0;
	for(size_t i = 0; i <= skip; ++i) {
		x = rng.rand(f);
	}

	return x;
}

TEST(Random, testJumpSmall) {
	Random rng, jrng;
	// 0, 1
	EXPECT_EQ(generate(rng), generate(jrng));
	EXPECT_EQ(generate(rng), generate(jrng));

	// 12, 13
	jrng.jump(10);
	EXPECT_EQ(generate(rng, 10), generate(jrng));
	EXPECT_EQ(generate(rng), generate(jrng));

	// 33, 34
	jrng.jump(18);
	EXPECT_EQ(generate(rng, 18), generate(jrng));
	EXPECT_EQ(generate(rng), generate(jrng));
}

uint32_t generate(Random &rng, size_t skip, size_t incr) {
	for(size_t i = 0; i < skip; i += incr) {
		rng.jump(incr);
	}

	basic_dist f;
	return rng.rand(f);
}

void test_nbit_with_mbit(int n, int m) {
	Random rng, jrng;

	size_t mbit = 2UL << (m - 1);
	size_t nbit = 2UL << (n - 1);

	for(int i = 0; i < 6; ++i) {
		jrng.jump(nbit);
		EXPECT_EQ(generate(rng, nbit, mbit), generate(jrng));
	}
}

TEST(Random, testJump) {
	test_nbit_with_mbit(8, 1);
	test_nbit_with_mbit(16, 8);
	test_nbit_with_mbit(32, 16);
	test_nbit_with_mbit(64, 32);
}

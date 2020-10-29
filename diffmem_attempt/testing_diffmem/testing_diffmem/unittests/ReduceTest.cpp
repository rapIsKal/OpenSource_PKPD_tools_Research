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
#include "common.hpp"
#include "reduce_sum.hpp"
#include <numeric>
#include <algorithm>

TEST(Reduce, vector_range_sum) {
	std::vector<double> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

	using It = std::vector<double>::const_iterator;
	auto reduce = reduce_sum(v, [](It it) { return *it; });
	static_assert( std::is_same<decltype(reduce), double>::value, "reduce_sum should return double");

	auto accum = std::accumulate(std::begin(v), std::end(v), 0.);
	static_assert( std::is_same<decltype(reduce), decltype(accum)>::value, "reduce_sum should return same type as std::accumulate");
	EXPECT_EQ(reduce, accum);

	double sum = 0.0;
	for(auto x : v)  sum += x;
	EXPECT_EQ(reduce, sum);
}

TEST(Reduce, vector_index_sum) {
	std::vector<double> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

	auto reduce = reduce_sum(size_t(0), v.size(), [&v](size_t i) { return v[i] * i; });
	static_assert( std::is_same<decltype(reduce), double>::value, "reduce_sum should return double");

	double sum = 0.0;
	for(size_t i = 0; i < v.size(); ++i)
		sum += v[i] * i;
	EXPECT_EQ(reduce, sum);
}


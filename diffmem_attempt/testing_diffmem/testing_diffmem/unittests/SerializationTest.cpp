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
#include "Serialization.hpp"
#include <fstream>

struct SomeData {
	int32_t id[3];

	template <class Archive>
	void save(Archive &ar) const {
		ar(id);
	}

	template <class Archive>
	void load(Archive &ar) {
		ar(id);
	}

	int32_t sum() const {
		return id[0] + id[1] + id[2];
	}
};

static Vecd test(double a, Vecd & y) {
	return a * y;
}

TEST(Serialization, testBuffer) {
	using namespace serialization;

	Buffer buffer;
	BufferOutputArchive archive(buffer);

	Vecd myData(10);
	myData << 1,2,3,4,5,6,7,8,9,0;
	pack(archive, 5.0, myData);

	SomeData sd = { {1,2,3} };
	archive(sd);

	BufferInputArchive iarchive(buffer);

	Vecd other = call(test, iarchive);
	EXPECT_TRUE(other == 5*myData);

	EXPECT_EQ(call(&SomeData::sum, iarchive), 6);
}

TEST(Serialization, testBinary) {
	using namespace serialization;

	std::ofstream os("out.cereal", std::ios::binary);
	cereal::BinaryOutputArchive archive(os);

	Vecd myData(10);
	myData << 1,2,3,4,5,6,7,8,9,0;
	pack(archive, 5.0, myData);

	SomeData sd = { {1,2,3} };
	archive(sd);

	os.close();
	std::ifstream is("out.cereal", std::ios::binary);
	cereal::BinaryInputArchive iarchive(is);

	Vecd other = call(test, iarchive);
	EXPECT_TRUE(other == 5*myData);

	EXPECT_EQ(call(&SomeData::sum, iarchive), 6);
}

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
#include "SmallVector.hpp"
#include <cstdarg>

class Constructable {
private:
	static int numConstructorCalls;
	static int numDestructorCalls;
	static int numAssignmentCalls;

	int value;

public:
	Constructable() : value(0) { ++numConstructorCalls; }

	Constructable(int val) : value(val) { ++numConstructorCalls; }

	Constructable(const Constructable &src) {
		value = src.value;
		++numConstructorCalls;
	}

	~Constructable() { ++numDestructorCalls; }

	Constructable &operator=(const Constructable &src) {
		value = src.value;
		++numAssignmentCalls;
		return *this;
	}

	int getValue() const { return abs(value); }

	static void reset() {
		numConstructorCalls = 0;
		numDestructorCalls = 0;
		numAssignmentCalls = 0;
	}

	static int getNumConstructorCalls() { return numConstructorCalls; }

	static int getNumDestructorCalls() { return numDestructorCalls; }

	friend bool operator==(const Constructable &c0, const Constructable &c1) {
		return c0.getValue() == c1.getValue();
	}
};

int Constructable::numConstructorCalls;
int Constructable::numDestructorCalls;
int Constructable::numAssignmentCalls;

typedef ::testing::Types<SmallVector<Constructable, 0>,
                         SmallVector<Constructable, 1>,
                         SmallVector<Constructable, 2>,
                         SmallVector<Constructable, 4>
                         > SmallVectorTestTypes;
TYPED_TEST_CASE(SmallVectorTest, SmallVectorTestTypes);

// Test fixture class
template <typename VectorT>
class SmallVectorTest : public testing::Test {
	public:
		void SetUp() { Constructable::reset(); }

		void assertEmpty() {
			// Size tests
			EXPECT_EQ(0u, v.size());
			EXPECT_TRUE(v.empty());

			// Iterator tests
			EXPECT_TRUE(v.begin() == v.end());
			EXPECT_TRUE(v.rbegin() == v.rend());
		}

		// Assert that theVector contains the specified values, in order.
		void assertValuesInOrder(size_t size, ...) {
			EXPECT_EQ(size, v.size());

			va_list ap;
			va_start(ap, size);
			for(size_t i = 0; i < size; ++i) {
				int value = va_arg(ap, int);
				EXPECT_EQ(value, v[i].getValue());
			}

			va_end(ap);
		}

		// Generate a sequence of values to initialize the vector.
		void makeSequence(int start, int end) {
			for(int i = start; i <= end; ++i) {
				v.push_back(Constructable(i));
			}
		}

	protected:
		VectorT v;
};

TYPED_TEST(SmallVectorTest, testEmpty) {
	this->assertEmpty();
	EXPECT_EQ(0, Constructable::getNumConstructorCalls());
	EXPECT_EQ(0, Constructable::getNumDestructorCalls());
}

TYPED_TEST(SmallVectorTest, testClear) {
	this->v.reserve(2);
	this->makeSequence(1, 2);
	this->v.clear();

	this->assertEmpty();
	EXPECT_EQ(4, Constructable::getNumConstructorCalls());
	EXPECT_EQ(4, Constructable::getNumDestructorCalls());
}

TYPED_TEST(SmallVectorTest, testResizeShrink) {
	this->v.reserve(3);
	this->makeSequence(1, 3);
	this->v.resize(1);

	this->assertValuesInOrder(1u, 1);
	EXPECT_EQ(6, Constructable::getNumConstructorCalls());
	EXPECT_EQ(5, Constructable::getNumDestructorCalls());
}

// Resize bigger test.
TYPED_TEST(SmallVectorTest, testResizeGrow) {
	this->v.resize(2);

	EXPECT_EQ(2, Constructable::getNumConstructorCalls());
	EXPECT_EQ(0, Constructable::getNumDestructorCalls());
	EXPECT_EQ(2u, this->v.size());
}

TYPED_TEST(SmallVectorTest, testResizeFill) {
	this->v.resize(3, Constructable(77));
	this->assertValuesInOrder(3u, 77, 77, 77);

	EXPECT_EQ(4, Constructable::getNumConstructorCalls());
	EXPECT_EQ(1, Constructable::getNumDestructorCalls());
}

// Overflow past fixed size.
TYPED_TEST(SmallVectorTest, testOverflow) {
	// Push more elements than the fixed size.
	this->makeSequence(1, 10);

	// Test size and values.
	EXPECT_EQ(10u, this->v.size());
	for(int i = 0; i < 10; ++i) {
		EXPECT_EQ(i + 1, this->v[i].getValue());
	}

	// Now resize back to fixed size.
	this->v.resize(1);

	this->assertValuesInOrder(1u, 1);
}

TYPED_TEST(SmallVectorTest, testAppend) {
	TypeParam otherVector;
	otherVector.push_back(2);
	otherVector.push_back(3);

	this->v.push_back(Constructable(1));
	this->v.append(otherVector.begin(), otherVector.end());

	this->assertValuesInOrder(3u, 1, 2, 3);
}

TYPED_TEST(SmallVectorTest, testAssign) {
	this->v.push_back(Constructable(1));
	this->v.assign(2, Constructable(77));
	this->assertValuesInOrder(2u, 77, 77);
}

TYPED_TEST(SmallVectorTest, testErase) {
	this->makeSequence(1, 3);
	this->v.erase(this->v.begin());
	this->assertValuesInOrder(2u, 2, 3);
}

TYPED_TEST(SmallVectorTest, testEraseRange) {
	this->makeSequence(1, 3);
	this->v.erase(this->v.begin(), this->v.begin() + 2);
	this->assertValuesInOrder(1u, 3);
}

TYPED_TEST(SmallVectorTest, testInsert) {
	this->makeSequence(1, 3);
	typename TypeParam::iterator I =
			this->v.insert(this->v.begin() + 1, Constructable(77));
	EXPECT_EQ(this->v.begin() + 1, I);
	this->assertValuesInOrder(4u, 1, 77, 2, 3);
}

TYPED_TEST(SmallVectorTest, testInsertRange) {
	Constructable arr[3] = { Constructable(77), Constructable(77), Constructable(77) };

	this->makeSequence(1, 3);
	typename TypeParam::iterator I = this->v.insert(this->v.begin() + 1, arr, arr + 3);
	EXPECT_EQ(this->v.begin() + 1, I);
	this->assertValuesInOrder(6u, 1, 77, 77, 77, 2, 3);

	// Insert at end.
	I = this->v.insert(this->v.end(), arr, arr + 3);
	EXPECT_EQ(this->v.begin() + 6, I);
	this->assertValuesInOrder(9u, 1, 77, 77, 77, 2, 3, 77, 77, 77);

	// Empty insert.
	EXPECT_EQ(this->v.end(),
			this->v.insert(this->v.end(), this->v.begin(), this->v.begin()));
	EXPECT_EQ(this->v.begin() + 1,
			this->v.insert(this->v.begin() + 1, this->v.begin(), this->v.begin()));
}

struct alignas(32) A {};

TEST(SmallVectorTest, alignment) {
	SmallVector<A, 10> v(10);
	for(auto & x : v) {
		EXPECT_EQ(__alignof__(x), alignof(A));
	}
}

TEST(SmallVectorTest, initializer) {
	SmallVector<int, 10> v = {0, 1, 2, 3, 4, 5};

	EXPECT_EQ(v.size(), 6UL);
	for(size_t i = 0; i < v.size(); ++i)
		EXPECT_EQ(static_cast<int>(i), v[i]);
}

TEST(SmallVectorTest, copyconstruct) {
	SmallVector<int, 10> v = {0, 1, 2, 3, 4, 5};
	auto w = v;

	EXPECT_EQ(v.size(), w.size());
	for(size_t i = 0; i < v.size(); ++i)
		EXPECT_EQ(w[i], v[i]);
}

TEST(SmallVectorTest, moveconstruct_small) {
	SmallVector<int, 10> v = {0, 1, 2, 3, 4, 5};
	auto w = std::move(v);

	EXPECT_EQ(w.size(), 6UL);
	for(size_t i = 0; i < w.size(); ++i)
		EXPECT_EQ(static_cast<int>(i), w[i]);

	EXPECT_TRUE(v.empty()); // shouldn't touch v after moving
	EXPECT_TRUE(v.isSmall()); // shouldn't touch v after moving
}

TEST(SmallVectorTest, moveconstruct_big) {
	SmallVector<int, 10> v(15);
	auto w = std::move(v);

	EXPECT_EQ(w.size(), 15UL);
	EXPECT_TRUE(v.empty()); // shouldn't touch v after moving
	EXPECT_TRUE(v.isSmall()); // shouldn't touch v after moving
}

TEST(SmallVectorTest, moveassign_small_from_bigger) {
	SmallVector<int, 10> v(7);
	SmallVector<int, 5> w(5);
	w = std::move(v);

	EXPECT_EQ(w.size(), 7UL);
	EXPECT_FALSE(w.isSmall());

	EXPECT_TRUE(v.empty()); // shouldn't touch v after moving
	EXPECT_TRUE(v.isSmall()); // shouldn't touch v after moving
}

TEST(SmallVectorTest, moveassign_small_from_smaller) {
	SmallVector<int, 5> v(5);
	SmallVector<int, 10> w(7);
	w = std::move(v);

	EXPECT_EQ(w.size(), 5UL);
	EXPECT_TRUE(w.isSmall());

	EXPECT_TRUE(v.empty()); // shouldn't touch v after moving
	EXPECT_TRUE(v.isSmall()); // shouldn't touch v after moving
}

TEST(SmallVectorTest, isSmall) {
	SmallVector<int, 10> v(10);
	EXPECT_TRUE(v.isSmall());
	v.push_back(1);
	EXPECT_FALSE(v.isSmall());
	v.pop_back();
	EXPECT_FALSE(v.isSmall());
}

const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const @"bool" = @import("bool.zig");
const all = @"bool".all;
const util = @import("util.zig");

/// Scalar data structure for floating point computation
pub fn Type(comptime Float: type) type {
    return FloatType(1, Float);
}

/// Scalar data structure for floating point computation using SIMD
/// This is meant for internal use, for scalar computations in userspace use 'Type()' instead.
fn FloatType(comptime size: ?usize, comptime Float: type) type {
    { // asserts
        if (@typeInfo(Float) != .Float) @compileError("'Float' must be a float");
        if (size == 0) @compileError("size must be > 0");
    }
    return struct {
        pub const Element = @This();
        pub const simd_size = size orelse (std.simd.suggestVectorLength(Float) orelse 1);
        const Bool = @"bool".Type(simd_size);

        f: @Vector(simd_size, Float),

        ///
        pub fn SimdType(comptime _size: ?usize) type {
            return FloatType(_size, Float);
        }

        /// the '0' element
        /// this is the default element in a sparse struct
        pub const zero = Element{ .f = @splat(0) };

        /// the '1' element
        pub const one = Element{ .f = @splat(1) };

        /// element isomorph to 'p/q'
        pub fn from(p: isize, q: usize) Element {
            const p_: Float = @floatFromInt(p);
            const q_: Float = @floatFromInt(q);
            return .{ .f = @splat(p_ / q_) };
        }

        fn Deref(a: type) type {
            const A = util.Deref(a);
            if (A.SimdType(1) != SimdType(1)) @compileError("a must be a SimdType of Element");
            return A;
        }

        /// a == b
        pub fn eq(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f == b.f);
        }

        /// a < b
        pub fn lt(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f < b.f);
        }

        /// a <= b
        pub fn lte(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f <= b.f);
        }

        /// a > b
        pub fn gt(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f > b.f);
        }

        /// a >= b
        pub fn gte(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f >= b.f);
        }

        /// a != b
        pub fn neq(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f != b.f);
        }

        /// a
        pub fn id(a: anytype) Deref(@TypeOf(a)).Element {
            return if (@typeInfo(@TypeOf(a)) == .Pointer) a.* else a;
        }

        /// a + b
        pub fn add(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Element {
            return .{ .f = a.f + b.f };
        }

        /// -a
        pub fn neg(a: anytype) Deref(@TypeOf(a)).Element {
            return .{ .f = -a.f };
        }

        /// a - b
        pub fn sub(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Element {
            return .{ .f = a.f - b.f };
        }

        /// a * b
        pub fn mul(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Element {
            return .{ .f = a.f * b.f };
        }

        /// 1 / a
        pub fn inv(a: anytype) !Deref(@TypeOf(a)).Element {
            if (!all(a.neq(Deref(@TypeOf(a)).Element.zero))) return error.DivisionByZero;
            return .{ .f = one.f / a.f };
        }

        /// a / b
        pub fn div(a: anytype, b: Deref(@TypeOf(a)).Element) !Deref(@TypeOf(a)).Element {
            if (!all(b.neq(Deref(@TypeOf(a)).Element.zero))) return error.DivisionByZero;
            return .{ .f = a.f / b.f };
        }

        /// min(a, b)
        pub fn min(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Element {
            return .{ .f = @min(a.f, b.f) };
        }

        /// max(a, b)
        pub fn max(a: anytype, b: Deref(@TypeOf(a)).Element) Deref(@TypeOf(a)).Element {
            return .{ .f = @max(a.f, b.f) };
        }

        /// |a|
        pub fn abs(a: anytype) Deref(@TypeOf(a)).Element {
            return .{ .f = @abs(a.f) };
        }

        /// sqrt(a)
        pub fn sqrt(a: anytype) !Deref(@TypeOf(a)).Element {
            if (!all(a.gte(Deref(@TypeOf(a)).Element.zero))) return error.SqrtOfNegative;
            return .{ .f = @sqrt(a.f) };
        }

        /// splat Element a of size 1 to simd_size
        pub fn simdSplat(a: SimdType(1)) Element {
            return Element{ .f = @splat(@bitCast(a.f)) };
        }

        pub fn reduceAdd(a: anytype) SimdType(1) {
            return .{ .f = @bitCast(@reduce(.Add, a.f)) };
        }

        pub fn reduceMul(a: anytype) SimdType(1) {
            return .{ .f = @bitCast(@reduce(.Mul, a.f)) };
        }

        pub fn reduceMin(a: anytype) SimdType(1) {
            return .{ .f = @bitCast(@reduce(.Min, a.f)) };
        }

        pub fn reduceMax(a: anytype) SimdType(1) {
            return .{ .f = @bitCast(@reduce(.Max, a.f)) };
        }
    };
}

test "float" {
    const float_types = .{ f16, f32, f64, f128 };
    inline for (float_types) |fx| {
        const F = Type(fx);
        try expect(F.zero.eq(F.from(0, 1)));
        try expect(F.one.eq(F.from(1, 1)));
        try expect(F.zero.neq(F.one));

        const two = F.one.add(F.one);
        // 1 - 2^-1 = 1 / 2
        try expect(F.one.sub(try two.inv()).eq(try F.one.div(two)));
        // sqrt(2 * 2) = |-2|
        try expect((try two.mul(two).sqrt()).eq(two.neg().abs()));
    }
}

test "float simd" {
    const simd_sizes = .{ 1, null };
    inline for (simd_sizes) |simd| {
        const float_types = .{ f16, f32, f64 }; //TODO: add f128, crashes in zig 0.13.0
        inline for (float_types) |fx| {
            const F = Type(fx);
            const SimdF = F.SimdType(simd);

            const a = SimdF.from(-1, 1);
            var b = SimdF.from(3, 1);
            b.f[0] = F.zero.f[0];
            const c = SimdF.from(2, 1);
            try expect(!all(a.add(b).eq(c)));
            try expect(all(a.add(b).neq(c)) == (SimdF.simd_size == 1));

            try expect(a.add(b).reduceMax().eq(F.from(if (SimdF.simd_size > 1) 2 else -1, 1)));
        }
    }
}

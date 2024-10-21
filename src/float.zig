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
/// operations (add, mul, eq, ...) must operate on any SIMD version
fn FloatType(comptime size: ?usize, comptime Float: type) type {
    comptime {
        if (size == 0) @compileError("size must be > 0 or null");
        if (@typeInfo(Float) != .Float)
            @compileError("expcected float, found " ++ @typeName(Float));
    }
    return struct {
        pub const Element = @This();
        pub const simd_size = size orelse (std.simd.suggestVectorLength(Float) orelse 1);
        const Bool = @"bool".Type(simd_size);

        f: @Vector(simd_size, Float),

        /// returns a SIMD version with the given size of the current type
        /// mainly for internal use
        pub fn SimdType(comptime _size: ?usize) type {
            return FloatType(_size, Float);
        }

        /// for internal use
        /// removes pointer type and asserts that the result is an Element or SIMD version of it
        fn Deref(a: type) type {
            const A = util.Deref(a);
            if (A.SimdType(1) != SimdType(1))
                @compileError("a must be a SimdType of Element, found " ++ @typeName(A));
            return A;
        }

        /// for internal use
        /// return Element with all SIMD entries set to `a`
        pub fn simdSplat(a: SimdType(1)) Element {
            return Element{ .f = @splat(@bitCast(a.f)) };
        }

        /// for internal use
        /// reduce SIMD element a to a single value
        /// op is expected to be the corresponding operator
        pub fn simdReduce(a: anytype, comptime op: anytype) Deref(@TypeOf(a)).SimdType(1) {
            comptime {
                const b: Element = undefined;
                assert(@TypeOf(@call(.auto, op, .{ b, b })) == Element);
            }
            return switch (op) {
                add => .{ .f = @bitCast(@reduce(.Add, a.f)) },
                mul => .{ .f = @bitCast(@reduce(.Mul, a.f)) },
                min => .{ .f = @bitCast(@reduce(.Min, a.f)) },
                max => .{ .f = @bitCast(@reduce(.Max, a.f)) },
                else => @compileError("reduce operation not supported"),
            };
        }

        /// the `0` element
        pub const zero = Element{ .f = @splat(0) };

        /// the `1` element
        pub const one = Element{ .f = @splat(1) };

        /// element given by the fraction `p/q`
        pub fn from(p: isize, q: usize) Element {
            const p_: Float = @floatFromInt(p);
            const q_: Float = @floatFromInt(q);
            return .{ .f = @splat(p_ / q_) };
        }

        /// a == b
        pub fn eq(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f == b.f);
        }

        /// a < b
        pub fn lt(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f < b.f);
        }

        /// a <= b
        pub fn lte(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f <= b.f);
        }

        /// a > b
        pub fn gt(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f > b.f);
        }

        /// a >= b
        pub fn gte(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f >= b.f);
        }

        /// a != b
        pub fn neq(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)).Bool {
            return @bitCast(a.f != b.f);
        }

        /// a
        pub fn id(a: anytype) Deref(@TypeOf(a)) {
            return if (@typeInfo(@TypeOf(a)) == .Pointer) a.* else a;
        }

        /// a + b
        pub fn add(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)) {
            return .{ .f = a.f + b.f };
        }

        /// -a
        pub fn neg(a: anytype) Deref(@TypeOf(a)) {
            return .{ .f = -a.f };
        }

        /// a - b
        pub fn sub(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)) {
            return .{ .f = a.f - b.f };
        }

        /// a * b
        pub fn mul(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)) {
            return .{ .f = a.f * b.f };
        }

        /// 1 / a
        pub fn inv(a: anytype) !Deref(@TypeOf(a)) {
            if (!all(a.neq(Deref(@TypeOf(a)).zero))) return error.DivisionByZero;
            return .{ .f = one.f / a.f };
        }

        /// a / b
        pub fn div(a: anytype, b: Deref(@TypeOf(a))) !Deref(@TypeOf(a)) {
            if (!all(b.neq(Deref(@TypeOf(a)).zero))) return error.DivisionByZero;
            return .{ .f = a.f / b.f };
        }

        /// min(a, b)
        pub fn min(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)) {
            return .{ .f = @min(a.f, b.f) };
        }

        /// max(a, b)
        pub fn max(a: anytype, b: Deref(@TypeOf(a))) Deref(@TypeOf(a)) {
            return .{ .f = @max(a.f, b.f) };
        }

        /// |a|
        pub fn abs(a: anytype) Deref(@TypeOf(a)) {
            return .{ .f = @abs(a.f) };
        }

        /// sqrt(a)
        pub fn sqrt(a: anytype) !Deref(@TypeOf(a)) {
            if (!all(a.gte(Deref(@TypeOf(a)).zero))) return error.SqrtOfNegative;
            return .{ .f = @sqrt(a.f) };
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

            const d = a.add(b).simdReduce(SimdF.max);
            const e = F.from(if (SimdF.simd_size > 1) 2 else -1, 1);
            try expect(d.eq(e));
        }
    }
}

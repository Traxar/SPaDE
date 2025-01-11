//! Note: these functions have to work for @Vector inputs as well as scalar inputs
//! Note: dereferencing not needed as these functions are not field decls
const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;

const simd = @import("simd.zig");

pub fn @"and"(a: anytype, b: anytype) @TypeOf(a) {
    if (@TypeOf(a) != @TypeOf(b)) @compileError("exptected matching types");
    if (@TypeOf(a) == bool) {
        return a and b;
    } else {
        return @select(bool, a, b, a);
    }
}

pub fn @"or"(a: anytype, b: anytype) @TypeOf(a) {
    if (@TypeOf(a) != @TypeOf(b)) @compileError("exptected matching types");
    if (@TypeOf(a) == bool) {
        return a or b;
    } else {
        return @select(bool, a, a, b);
    }
}

pub fn not(a: anytype) @TypeOf(a) {
    const A = @TypeOf(a);
    if (A == bool) {
        return !a;
    } else {
        return @select(bool, a, @as(A, @splat(false)), @as(A, @splat(true)));
    }
}

test "boolean" {
    const a: @Vector(4, bool) = .{ false, false, true, true };
    const b: @Vector(4, bool) = .{ false, true, false, true };

    try expect(@reduce(.And, @"and"(a, b) == @as(@Vector(4, bool), .{ false, false, false, true })));
    try expect(@reduce(.And, @"or"(a, not(b)) == @as(@Vector(4, bool), .{ true, false, true, true })));
}

pub fn eq(a: anytype, b: anytype) @TypeOf(a == b) {
    return a == b;
}

pub fn neq(a: anytype, b: anytype) @TypeOf(a != b) {
    return a != b;
}

pub fn lt(a: anytype, b: anytype) @TypeOf(a < b) {
    return a < b;
}

pub fn lte(a: anytype, b: anytype) @TypeOf(a <= b) {
    return a <= b;
}

pub fn gt(a: anytype, b: anytype) @TypeOf(a > b) {
    return a > b;
}

pub fn gte(a: anytype, b: anytype) @TypeOf(a >= b) {
    return a >= b;
}

test "comparisons" {
    const a: u13 = 123;
    const b: u13 = 234;

    try expect(eq(a, a));
    try expect(!eq(a, b));

    try expect(!neq(a, a));
    try expect(neq(a, b));

    try expect(!lt(a, a));
    try expect(lt(a, b));

    try expect(lte(a, a));
    try expect(lte(a, b));

    try expect(!gt(a, a));
    try expect(!gt(a, b));

    try expect(gte(a, a));
    try expect(!gte(a, b));
}

pub fn id(a: anytype) @TypeOf(a) {
    return a;
}

pub fn add(a: anytype, b: anytype) @TypeOf(a + b) {
    return a + b;
}

pub fn neg(a: anytype) @TypeOf(-a) {
    return -a;
}

pub fn sub(a: anytype, b: anytype) @TypeOf(a - b) {
    return a - b;
}

pub fn mul(a: anytype, b: anytype) @TypeOf(a * b) {
    return a * b;
}

test {
    try expect(mul(2, 4) == 8);
}

pub fn inv(a: anytype) !@TypeOf(1 / a) {
    if (@typeInfo(@TypeOf(a)) == .Vector) {
        if (@reduce(.Or, a == 0)) return error.DivisionByZero;
        return @as(@TypeOf(a), @splat(1)) / a;
    } else {
        if (a == 0) return error.DivisionByZero;
        return 1 / a;
    }
}

pub fn div(a: anytype, b: anytype) !@TypeOf(a / b) {
    if (@typeInfo(@TypeOf(b)) == .Vector) {
        if (@reduce(.Or, b == 0)) return error.DivisionByZero;
    } else {
        if (b == 0) return error.DivisionByZero;
    }
    return a / b;
}

pub fn min(a: anytype, b: anytype) @TypeOf(@min(a, b)) {
    return @min(a, b);
}

/// max(a, b)
pub fn max(a: anytype, b: anytype) @TypeOf(@max(a, b)) {
    return @max(a, b);
}

/// |a|
pub fn abs(a: anytype) @TypeOf(@abs(a)) {
    return @abs(a);
}

/// sqrt(a)
pub fn sqrt(a: anytype) !@TypeOf(@sqrt(a)) {
    if (@typeInfo(@TypeOf(a)) == .Vector) {
        if (@reduce(.Or, a < 0)) return error.DivisionByZero;
    } else {
        if (a < 0) return error.DivisionByZero;
    }
    return @sqrt(a);
}

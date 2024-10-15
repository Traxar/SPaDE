const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;

pub const Majority = enum {
    row,
    col,

    pub fn other(maj: Majority) Majority {
        return switch (maj) {
            .row => .col,
            .col => .row,
        };
    }
};

pub inline fn isElement(type_in_question: type) bool {
    comptime {
        return std.meta.hasFn(type_in_question, "SimdType");
    }
}

pub fn ArgsType(comptime step: usize, comptime Args: type) type {
    const info_args = @typeInfo(Args).Struct;
    var fields: [info_args.fields.len]std.builtin.Type.StructField = undefined;
    for (info_args.fields, &fields) |src, *dest| {
        const Arg = (if (isElement(src.type)) src.type else src.type.Element).SimdType(step);
        dest.* = .{
            .alignment = @alignOf(Arg),
            .default_value = null,
            .is_comptime = false,
            .name = src.name,
            .type = Arg,
        };
    }
    return @Type(std.builtin.Type{ .Struct = .{
        .fields = &fields,
        .decls = &.{},
        .is_tuple = info_args.is_tuple,
        .layout = info_args.layout,
    } });
}

pub inline fn argsSimdSize(comptime Args: type) usize {
    comptime {
        const info_args = @typeInfo(Args).Struct;
        const max = std.math.maxInt(usize);
        var min_simd_size = max;
        for (info_args.fields) |field| {
            const Arg = (if (isElement(field.type)) field.type else field.type.Element).SimdType(null);
            min_simd_size = @min(min_simd_size, Arg.simd_size);
        }
        assert(min_simd_size < max);
        return min_simd_size;
    }
}

pub inline fn argsMajority(comptime Args: type) ?Majority {
    comptime {
        const info_args = @typeInfo(Args).Struct;
        var maj: ?Majority = null;
        for (info_args.fields) |field| {
            if (isElement(field.type)) continue;
            maj = maj orelse field.type.major;
            if (maj != field.type.major) @compileError("majority does not match");
        }
        return maj;
    }
}

/// asserts that all argument dimensions are the same and returns them in form of a matrix with no values
pub fn argsDimensions(args: anytype) struct { rows: usize, cols: usize } {
    const info_args = @typeInfo(@TypeOf(args)).Struct;
    var r: ?usize = null;
    var c: ?usize = null;
    inline for (info_args.fields) |field| {
        if (isElement(field.type)) continue;
        const arg = @field(args, field.name);
        r = r orelse arg.rows;
        c = c orelse arg.cols;
        if (r != arg.rows or c != arg.cols) unreachable;
    }
    return .{ .rows = r orelse 1, .cols = c orelse 1 };
}

///set arguments given as element
pub fn argsPrep(comptime step: usize, args: anytype) ArgsType(step, @TypeOf(args)) {
    var a: ArgsType(step, @TypeOf(args)) = undefined;
    const info_args = @typeInfo(@TypeOf(args)).Struct;
    inline for (info_args.fields) |field| {
        if (!isElement(field.type)) continue;
        const arg = @field(args, field.name);
        @field(a, field.name) = @TypeOf(@field(a, field.name)).simdSplat(arg);
    }
    return a;
}

test "args" {
    const F = @import("float.zig").Type(f32);
    const G = @import("float.zig").Type(f64);
    const n = 4;
    const args = .{ F.zero, G.one };
    const simd_args = argsPrep(n, args);

    try expect(@TypeOf(simd_args[0]) == F.SimdType(n));
    try expect(@TypeOf(simd_args[1]) == G.SimdType(n));
    try expect(@reduce(.And, simd_args[0].eq(F.SimdType(n).simdSplat(F.zero))));
    try expect(@reduce(.And, simd_args[1].eq(G.SimdType(n).simdSplat(G.one))));
}

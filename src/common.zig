const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;

/// matrices are stored `majority`-wise in memory
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

/// return `simd` version of arguments 'Args'.
/// fields of `Args` must has a constant decl `Element` giving the underlying type.
pub fn ArgsType(comptime simd: usize, comptime Args: type) type {
    const info_args = @typeInfo(Args).Struct;
    var fields: [info_args.fields.len]std.builtin.Type.StructField = undefined;
    for (info_args.fields, &fields) |src, *dest| {
        const Arg = src.type.Element.SimdType(simd);
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

/// find smallest SIMD size suggested by the fields of `Args`
pub inline fn argsSimdSize(comptime Args: type) usize {
    comptime {
        const info_args = @typeInfo(Args).Struct;
        var simd_size_min: ?usize = null;
        for (info_args.fields) |field| {
            const suggested = field.type.Element.SimdType(null).simd_size;
            simd_size_min = if (simd_size_min) |min| @min(min, suggested) else suggested;
        }
        return simd_size_min orelse 1;
    }
}

/// finds the common majority of the matrices in `Args`
pub inline fn argsMajority(comptime Args: type) ?Majority {
    comptime {
        const info_args = @typeInfo(Args).Struct;
        var maj: ?Majority = null;
        for (info_args.fields) |field| {
            if (field.type.Element == field.type) continue;
            maj = maj orelse field.type.major;
            if (maj != field.type.major) @compileError("majorities do not match");
        }
        return maj;
    }
}

/// finds the common matrix dimensions
pub fn argsDimensions(args: anytype) struct { rows: usize, cols: usize } {
    const info_args = @typeInfo(@TypeOf(args)).Struct;
    var r: ?usize = null;
    var c: ?usize = null;
    inline for (info_args.fields) |field| {
        if (field.type.Element == field.type) continue;
        const arg = @field(args, field.name);
        r = r orelse arg.rows;
        assert(r == arg.rows);
        c = c orelse arg.cols;
        assert(c == arg.cols);
    }
    return .{ .rows = r orelse 1, .cols = c orelse 1 };
}

///set all arguments given as element
pub fn argsPrep(comptime step: usize, args: anytype) ArgsType(step, @TypeOf(args)) {
    var a: ArgsType(step, @TypeOf(args)) = undefined;
    const info_args = @typeInfo(@TypeOf(args)).Struct;
    inline for (info_args.fields) |field| {
        if (field.type.Element != field.type) continue;
        const arg = @field(args, field.name);
        @field(a, field.name) = @TypeOf(@field(a, field.name)).simdSplat(arg);
    }
    return a;
}

test "args" {
    const F = @import("float.zig").Type(f32);
    const G = @import("float.zig").Type(f64);
    const Bool = @import("bool.zig");
    const n = 4;
    const args = .{ F.zero, G.one };
    const simd_args = argsPrep(n, args);

    try expect(@TypeOf(simd_args[0]) == F.SimdType(n));
    try expect(@TypeOf(simd_args[1]) == G.SimdType(n));
    try expect(Bool.all(simd_args[0].eq(F.SimdType(n).simdSplat(F.zero))));
    try expect(Bool.all(simd_args[1].eq(G.SimdType(n).simdSplat(G.one))));
}

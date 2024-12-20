const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const util = @import("util.zig");
const tensor = @import("tensor.zig");
const dimens = @import("dimens.zig");
const coords = @import("coords.zig");
const position = @import("position.zig");

/// replace tensor types by their element types
pub fn Type(AnyArgs: type) type {
    const info = @typeInfo(AnyArgs).Struct;
    var fields: [info.fields.len]std.builtin.Type.StructField = undefined;
    for (info.fields, &fields) |src, *dest| {
        const is_tensor = tensor.is(src.type);
        const Arg = if (is_tensor) src.type.Element else src.type;
        dest.* = .{
            .alignment = @alignOf(Arg),
            .default_value = if (is_tensor) null else src.default_value,
            .is_comptime = if (is_tensor) false else src.is_comptime,
            .name = src.name,
            .type = Arg,
        };
    }
    return @Type(std.builtin.Type{ .Struct = .{
        .fields = &fields,
        .decls = &.{},
        .is_tuple = info.is_tuple,
        .layout = info.layout,
    } });
}

test "Args Type" {
    const A = struct { tensor.Type(f32).Dense(&.{ 0, 1 }), tensor.Type(bool).Dense(&.{2}), isize };
    const Arg = Type(A);
    const arg: Arg = undefined;
    try expect(@TypeOf(arg[0]) == f32);
    try expect(@TypeOf(arg[1]) == bool);
    try expect(@TypeOf(arg[2]) == isize);
}

pub fn init(anyargs: anytype) Type(@TypeOf(anyargs)) {
    const AnyArgs = @TypeOf(anyargs);
    var res: Type(AnyArgs) = undefined;
    inline for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
        if (tensor.is(field_anyargs.type)) continue;
        @field(res, field_anyargs.name) = @field(anyargs, field_anyargs.name);
    }
    return res;
}

test "Args init" {
    const A = struct { tensor.Type(f32).Dense(&.{ 0, 1 }), tensor.Type(bool).Dense(&.{2}), isize };
    const a: A = .{ undefined, undefined, 123 };
    const arg = init(a);
    try expect(arg[2] == 123);
}

pub fn set(anyargs: anytype, args: *Type(@TypeOf(anyargs)), coord: []const usize) void {
    const AnyArgs = @TypeOf(anyargs);
    inline for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
        if (!tensor.is(field_anyargs.type)) continue;
        @field(args, field_anyargs.name) = @field(anyargs, field_anyargs.name).at(coord);
    }
}

pub inline fn collectDims(AnyArgs: type) []const usize {
    comptime {
        var res: []const usize = &.{};
        for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
            if (!tensor.is(field_anyargs.type)) continue;
            res = dimens._union(res, field_anyargs.type.dims);
        }
        return res;
    }
}

test "Args dims" {
    const A = struct { tensor.Type(f32).Dense(&.{ 0, 1 }), tensor.Type(bool).Dense(&.{2}), isize };
    const d = collectDims(A);
    try expect(d.len == 3);
    try expect(d[0] == 0);
    try expect(d[1] == 1);
    try expect(d[2] == 2);
}

pub fn collectSize(size: []usize, anyargs: anytype) void {
    const AnyArgs = @TypeOf(anyargs);
    inline for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
        coords.collectSize(size, @field(anyargs, field_anyargs.name));
    }
}

pub fn Return(f: anytype, AnyArgs: type, Base: type) type {
    return if (util.ErrorSet(f, Type(AnyArgs))) |Err| Err!Base else Base;
}

pub fn validInplace(res: anytype, anyargs: anytype) bool {
    const Res = util.Deref(@TypeOf(res));
    if (!tensor.is(Res)) @compileError("res must be a Tensor or *Tensor");
    if (Res.dims.len == 0) return true;
    const AnyArgs = @TypeOf(anyargs);
    inline for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
        const AnyArg = field_anyargs.type;
        if (tensor.is(AnyArg) and AnyArg.dims.len == Res.dims.len) {
            const anyarg = @field(anyargs, field_anyargs.name);
            if (anyarg.vals.ptr == res.vals.ptr) {
                for (Res.dims, AnyArg.dims) |r, a| {
                    if (r != a) return false;
                }
            }
        }
    }
    return true;
}

test "validInplace" {
    const ally = std.testing.allocator;
    const M = tensor.Type(f32).Dense(&.{ 0, 1 });
    const a = try M.init(&.{ 2, 2 }, ally);
    defer a.deinit(ally);
    const aT: tensor.Type(f32).Dense(&.{ 1, 0 }) = @bitCast(a);
    try expect(validInplace(a, .{ a, a, a }));
    try expect(!validInplace(a, .{ a, aT, a }));
}

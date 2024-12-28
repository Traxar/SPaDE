const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const util = @import("util.zig");
const tensor = @import("tensor.zig");
const Dims = @import("dims.zig").Type;
const coords = @import("coords.zig");
const position = @import("position.zig");

/// replace tensor types by their element types
fn Internal(AnyArgs: type) type {
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

test "Args" {
    const A = struct { tensor.Type(f32).Dense(&.{ 0, 1 }), tensor.Type(bool).Dense(&.{2}), isize };
    const Arg = Internal(A);
    const arg: Arg = undefined;
    try expect(@TypeOf(arg[0]) == f32);
    try expect(@TypeOf(arg[1]) == bool);
    try expect(@TypeOf(arg[2]) == isize);
}

pub fn Type(AnyArgs: type) type {
    return struct {
        const Args = @This();
        args: Internal(AnyArgs),

        pub fn init(anyargs: AnyArgs) Args {
            var res: Args = undefined;
            inline for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
                if (tensor.is(field_anyargs.type)) continue;
                @field(res.args, field_anyargs.name) = @field(anyargs, field_anyargs.name);
            }
            return res;
        }

        pub fn set(args: *Args, anyargs: AnyArgs, coord: []const usize) void {
            inline for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
                if (!tensor.is(field_anyargs.type)) continue;
                @field(args.args, field_anyargs.name) = @field(anyargs, field_anyargs.name).at(coord);
            }
        }

        pub const dims = dims: {
            var res = Dims.from(&.{});
            for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
                if (!tensor.is(field_anyargs.type)) continue;
                res = res.unite(field_anyargs.type.dims);
            }
            break :dims res;
        };

        pub fn Return(f: anytype, Base: type) type {
            return if (util.ErrorSet(util.ReturnType(f, Internal(AnyArgs)))) |Err| Err!Base else Base;
        }

        pub fn call(args: Args, f: anytype) util.ReturnType(f, Internal(AnyArgs)) {
            return @call(.auto, f, args.args);
        }
    };
}

// test "Args init" {
//     const A = struct { tensor.Type(f32).Dense(&.{ 0, 1 }), tensor.Type(bool).Dense(&.{2}), isize };
//     const a: A = .{ undefined, undefined, 123 };
//     const arg = init(a);
//     try expect(arg[2] == 123);
// }

// test "Args dims" {
//     const A = struct { tensor.Type(f32).Dense(&.{ 0, 1 }), tensor.Type(bool).Dense(&.{2}), isize };
//     const d = collectDims(A);
//     try expect(d.len == 3);
//     try expect(d.ptr[0] == 0);
//     try expect(d.ptr[1] == 1);
//     try expect(d.ptr[2] == 2);
// }

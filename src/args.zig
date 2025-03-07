const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const util = @import("util.zig");
const Dims = @import("dims.zig").Type;
const tensor = @import("tensor.zig");

/// replace tensor types by their element types
fn Internal(AnyArgs: type) type {
    const info = @typeInfo(AnyArgs).@"struct";
    var fields: [info.fields.len]std.builtin.Type.StructField = undefined;
    for (info.fields, &fields) |src, *dest| {
        const is_tensor = tensor.is(src.type);
        const Arg = if (is_tensor) src.type.Element else src.type;
        dest.* = .{
            .alignment = @alignOf(Arg),
            .default_value_ptr = if (is_tensor) null else src.default_value_ptr,
            .is_comptime = if (is_tensor) false else src.is_comptime,
            .name = src.name,
            .type = Arg,
        };
    }
    return @Type(std.builtin.Type{ .@"struct" = .{
        .fields = &fields,
        .decls = &.{},
        .is_tuple = info.is_tuple,
        .layout = info.layout,
    } });
}

test "Internal" {
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
        vals: Internal(AnyArgs),

        /// set all non tensor values of the result to those of `anyargs`
        pub fn init(anyargs: AnyArgs) Args {
            var res: Args = undefined;
            inline for (@typeInfo(AnyArgs).@"struct".fields) |field_anyargs| {
                if (tensor.is(field_anyargs.type)) continue;
                @field(res.vals, field_anyargs.name) = @field(anyargs, field_anyargs.name);
            }
            return res;
        }

        /// set all tensor values of `args` to those of `anyargs` at coordinates `coord`
        pub fn set(args: *Args, anyargs: AnyArgs, coord: []const usize) void {
            inline for (@typeInfo(AnyArgs).@"struct".fields) |field_anyargs| {
                if (!tensor.is(field_anyargs.type)) continue;
                @field(args.vals, field_anyargs.name) = @field(anyargs, field_anyargs.name).at(coord);
            }
        }

        pub const dims = _: {
            var res = Dims.from(&.{});
            for (@typeInfo(AnyArgs).@"struct".fields) |field_anyargs| {
                if (!tensor.is(field_anyargs.type)) continue;
                res = res.unite(field_anyargs.type.dims);
            }
            break :_ res;
        };

        /// if function `f` errors, wrap `Base` with its error
        pub fn ErrorWrap(f: anytype, Base: type) type {
            return if (util.ErrorSet(util.ReturnType(f, Internal(AnyArgs)))) |Err| Err!Base else Base;
        }

        /// call function `f` on arguments `args`
        pub fn call(args: Args, f: anytype) util.ReturnType(f, Internal(AnyArgs)) {
            return @call(.auto, f, args.vals);
        }
    };
}

test "Args init/set" {
    const ally = std.testing.allocator;
    const M = tensor.Type(f32).Dense(&.{ 0, 1 });
    const V = tensor.Type(bool).Dense(&.{2});
    const A = struct { M, V, isize };
    const a = A{ try M.init(&.{ 2, 2 }, ally), try V.init(&.{ 0, 0, 3 }, ally), 123 };
    defer {
        a[0].deinit(ally);
        a[1].deinit(ally);
    }
    a[0].set(&.{ 0, 0, 0 }, 1.0);
    a[1].set(&.{ 0, 0, 0 }, false);
    a[0].set(&.{ 1, 1, 1 }, 3.5);
    a[1].set(&.{ 1, 1, 1 }, true);

    var arg = Type(A).init(a);
    try expect(arg.vals[2] == 123);

    arg.set(a, &.{ 0, 0, 0 });
    try expect(arg.vals[0] == 1.0);
    try expect(arg.vals[1] == false);
    try expect(arg.vals[2] == 123);

    arg.set(a, &.{ 1, 1, 1 });
    try expect(arg.vals[0] == 3.5);
    try expect(arg.vals[1] == true);
    try expect(arg.vals[2] == 123);
}

test "Args dims" {
    const A = struct { tensor.Type(f32).Dense(&.{ 0, 1 }), tensor.Type(bool).Dense(&.{2}), isize };
    const d = Type(A).dims;
    try expect(d.len == 3);
    try expect(d.ptr[0] == 0);
    try expect(d.ptr[1] == 1);
    try expect(d.ptr[2] == 2);
}

test "Args ErrorWrap" {
    const op = @import("op.zig");
    const A = struct { f32, f32 };
    const Args = Type(A);
    try expect(Args.ErrorWrap(op.add, bool) == bool);
    try expect(Args.ErrorWrap(op.div, bool) != bool);
}

test "Args call" {
    const op = @import("op.zig");
    const A = struct { f32, f32 };
    const a = A{ 1, 2 };
    const args = Type(A).init(a);
    try expect(args.call(op.add) == 3);
}

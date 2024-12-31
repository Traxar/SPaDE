const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;

/// recurrsive vector struct
pub fn Vector(len: comptime_int, Element: type) type {
    return switch (@typeInfo(Element)) {
        .Struct => |s| _: {
            var res_fields: [s.fields.len]std.builtin.Type.StructField = undefined;
            for (s.fields, &res_fields) |field, *res_field| {
                res_field.* = .{
                    .alignment = 0,
                    .default_value = null,
                    .is_comptime = field.is_comptime,
                    .name = field.name,
                    .type = Vector(len, field.type),
                };
            }
            break :_ @Type(std.builtin.Type{
                .Struct = .{
                    .layout = s.layout,
                    .fields = &res_fields,
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = s.is_tuple,
                },
            });
        },
        .Vector => |v| if (len == 1) v.child else @Vector(len, v.child),
        else => if (len == 1) Element else @Vector(len, Element),
    };
}

test "Vector" {
    const A = struct { a: bool, b: u13, c: f32 };
    const C = Vector(3, A);
    const c: C = undefined;
    try expect(@TypeOf(c.a) == @Vector(3, bool));
    try expect(@TypeOf(c.b) == @Vector(3, u13));
    try expect(@TypeOf(c.c) == @Vector(3, f32));

    //try expect(A == Vector(1, C)); //watch out, this fails!
    try expect(Vector(1, A) == Vector(1, C));
}

/// retrieve the simd length of a recurrsive vector struct
pub fn length(Element: type) comptime_int {
    return switch (@typeInfo(Element)) {
        .Struct => |s| _: {
            var res: ?comptime_int = null;
            for (s.fields) |field| {
                const now = length(field.type);
                res = if (res) |sofar| sofar else now;
                if (res != now) @compileError("simd sizes do not match");
            }
            break :_ res.?;
        },
        .Vector => |v| v.len,
        else => 1,
    };
}

test "length" {
    const A = struct { a: bool, b: u13, c: f32 };
    const B = Vector(1, A);
    const C = Vector(3, A);
    try expect(length(A) == 1);
    try expect(length(B) == 1);
    try expect(length(C) == 3);
}

/// splat element into a recurrsive vector struct
pub fn splat(len: comptime_int, element: anytype) Vector(len, @TypeOf(element)) {
    assert(length(@TypeOf(element)) == 1);
    return switch (@typeInfo(@TypeOf(element))) {
        .Struct => |s| _: {
            var res: Vector(len, @TypeOf(element)) = undefined;
            inline for (s.fields) |field| {
                @field(res, field.name) = splat(len, @field(element, field.name));
            }
            break :_ res;
        },
        .Vector => @splat(@bitCast(element)),
        else => @splat(element),
    };
}

test "splat" {
    const A = struct { a: bool, b: u13, c: f32 };
    const a = A{ .a = true, .b = 123, .c = -0.5 };
    const c = splat(3, a);
    try expect(@TypeOf(c) == Vector(3, A));

    try expect(c.a[0] == true);
    try expect(c.a[1] == true);
    try expect(c.a[2] == true);

    try expect(c.b[0] == 123);
    try expect(c.b[1] == 123);
    try expect(c.b[2] == 123);

    try expect(c.c[0] == -0.5);
    try expect(c.c[1] == -0.5);
    try expect(c.c[2] == -0.5);
}

/// suggest simd length of a recurrsive vector struct
/// based on std heuristics
pub fn suggestLength(Element: type) comptime_int {
    return switch (@typeInfo(Element)) {
        .Struct => |s| _: {
            var res: ?comptime_int = null;
            for (s.fields) |field| {
                const now = suggestLength(field.type);
                res = if (res) |sofar| @min(sofar, now) else now;
            }
            break :_ res;
        },
        .Vector => |v| std.simd.suggestVectorLength(v.child),
        else => std.simd.suggestVectorLength(Element),
    } orelse 1;
}

test "suggestLength" {
    const A = struct { a: bool, b: u13, c: f32 };
    try expect(suggestLength(A) == suggestLength(f32));
}

pub fn As(V: type, Element: type) type {
    return Vector(length(V), Element);
}

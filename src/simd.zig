const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;

/// recurrsive vector struct
pub fn Vector(len: comptime_int, Element: type) type {
    assert(len > 0);
    if (len == 1) return Element; //? does this still make sense?
    return switch (@typeInfo(Element)) {
        .@"struct" => |s| _: {
            var res_fields: [s.fields.len]std.builtin.Type.StructField = undefined;
            for (s.fields, &res_fields) |field, *res_field| {
                res_field.* = .{
                    .alignment = 0,
                    .default_value_ptr = null,
                    .is_comptime = field.is_comptime,
                    .name = field.name,
                    .type = Vector(len, field.type),
                };
            }
            break :_ @Type(std.builtin.Type{
                .@"struct" = .{
                    .layout = s.layout,
                    .fields = &res_fields,
                    .decls = &[_]std.builtin.Type.Declaration{},
                    .is_tuple = s.is_tuple,
                },
            });
        },
        .array => |a| @Type(std.builtin.Type{
            .array = .{
                .child = Vector(len, a.child),
                .len = a.len,
                .sentinel_ptr = if (a.sentinel_ptr == null) null else &splat(len, a.sentinel_ptr.*),
            },
        }),
        .vector => |v| @Type(std.builtin.Type{
            .array = .{
                .child = Vector(len, v.child),
                .len = v.len,
                .sentinel_ptr = null,
            },
        }),
        else => @Vector(len, Element),
    };
}

test "Vector" {
    const A = struct { a: bool, b: u13, c: [4]f32 };
    const B = Vector(3, A);
    const b: B = undefined;
    try expect(@TypeOf(b.a) == @Vector(3, bool));
    try expect(@TypeOf(b.b) == @Vector(3, u13));
    try expect(@TypeOf(b.c) == [4]@Vector(3, f32));

    try expect(Vector(1, A) == A);
    try expect(Vector(1, B) == B);

    const C = Vector(2, B);
    const c: C = undefined;
    try expect(@TypeOf(c.a) == [3]@Vector(2, bool));
    try expect(@TypeOf(c.b) == [3]@Vector(2, u13));
    try expect(@TypeOf(c.c) == [4][3]@Vector(2, f32));
}

/// smallest simd length in `Element`, null if none are found
fn minVectorLength(Element: type) ?comptime_int {
    return switch (@typeInfo(Element)) {
        .@"struct" => |s| _: {
            var res: ?comptime_int = null;
            for (s.fields) |field| {
                if (minVectorLength(field.type)) |now|
                    res = if (res) |sofar| @min(sofar, now) else now;
            }
            break :_ res;
        },
        .array => |a| minVectorLength(a.child),
        .vector => |v| v.len,
        else => 1,
    };
}

test "minVectorLength" {
    const A = struct { a: bool, b: u13, c: f32 };
    try expect(minVectorLength(A) == 1);
    const B = struct { a: bool, b: @Vector(3, u13), c: f32 };
    try expect(minVectorLength(B) == 1);
    const C = Vector(4, B);
    try expect(minVectorLength(C) == 4);
}

/// returns `len` if `MaybeVector` is a result of `Vector(len , Element)` and null otherwise.
pub fn length(Element: type, MaybeVector: type) ?comptime_int {
    if (Element == MaybeVector) return 1;
    const len = minVectorLength(MaybeVector) orelse 1;
    if (Vector(len, Element) == MaybeVector) return len;
    return null;
}

test "length" {
    const A = struct { a: bool, b: u13, c: f32 };
    const B = struct { a: bool, b: @Vector(3, u13), c: f32 };
    const C = Vector(3, A);
    try expect(length(A, A) == 1);
    try expect(length(A, B) == null);
    try expect(length(A, C) == 3);
    try expect(length(B, B) == 1);
    try expect(length(C, C) == 1);
}

/// splat element into a recurrsive vector struct
pub fn splat(len: comptime_int, element: anytype) Vector(len, @TypeOf(element)) {
    if (len == 1) return element;
    return switch (@typeInfo(@TypeOf(element))) {
        .@"struct" => |s| _: {
            var res: Vector(len, @TypeOf(element)) = undefined;
            inline for (s.fields) |field| {
                @field(res, field.name) = splat(len, @field(element, field.name));
            }
            break :_ res;
        },
        .array, .vector => _: {
            var res: Vector(len, @TypeOf(element)) = undefined;
            inline for (element, 0..) |entry, i| {
                res[i] = splat(len, entry);
            }
            break :_ res;
        },
        else => @splat(element),
    };
}

test "splat" {
    const A = struct { a: bool, b: u13, c: [2]f32 };
    const a = A{ .a = true, .b = 123, .c = .{ -0.5, 0.3 } };
    const c = splat(3, a);
    try expect(@TypeOf(c) == Vector(3, A));

    try expect(c.a[0] == true);
    try expect(c.a[1] == true);
    try expect(c.a[2] == true);

    try expect(c.b[0] == 123);
    try expect(c.b[1] == 123);
    try expect(c.b[2] == 123);

    try expect(c.c[0][0] == -0.5);
    try expect(c.c[0][1] == -0.5);
    try expect(c.c[0][2] == -0.5);
    try expect(c.c[1][0] == 0.3);
    try expect(c.c[1][1] == 0.3);
    try expect(c.c[1][2] == 0.3);
}

/// suggest simd length of a recurrsive vector struct
/// based on std heuristics
pub fn suggestLength(Element: type) ?comptime_int {
    return switch (@typeInfo(Element)) {
        .@"struct" => |s| _: {
            var res: ?comptime_int = null;
            for (s.fields) |field| {
                if (suggestLength(field.type)) |now|
                    res = if (res) |sofar| @min(sofar, now) else now;
            }
            break :_ res;
        },
        .array => |a| suggestLength(a.child),
        .vector => |v| suggestLength(v.child),
        else => std.simd.suggestVectorLength(Element),
    };
}

test "suggestLength" {
    const A = struct { a: bool, b: u13, c: @Vector(2, f32) };
    try expect(suggestLength(A) == suggestLength(f32));
}

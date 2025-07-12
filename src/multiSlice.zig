const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;

/// recurrsive MultiPointer without functions
fn MultiPointer(Element: type) type {
    const info_E = @typeInfo(Element);
    return switch (info_E) {
        .@"struct" => |s| _: {
            var fields_MP: [s.fields.len]std.builtin.Type.StructField = undefined;
            for (&fields_MP, s.fields) |*field_MP, field_E| {
                field_MP.* = .{
                    .alignment = 0,
                    .default_value_ptr = null,
                    .is_comptime = field_E.is_comptime,
                    .name = field_E.name,
                    .type = MultiPointer(field_E.type),
                };
            }
            break :_ @Type(std.builtin.Type{ .@"struct" = .{
                .layout = .auto,
                .fields = &fields_MP,
                .decls = &[_]std.builtin.Type.Declaration{},
                .is_tuple = false,
            } });
        },
        .array => |a| [a.len]MultiPointer(a.child),
        .vector => |v| [v.len]MultiPointer(v.child),
        else => [*]Element,
    };
}

test MultiPointer {
    const A = struct { a: bool, b: @Vector(3, u16), c: [4]f32 };
    const B = MultiPointer(A);
    const b: B = undefined;
    try expect(@TypeOf(b.a) == [*]bool);
    try expect(@TypeOf(b.b) == [3][*]u16);
    try expect(@TypeOf(b.c) == [4][*]f32);
}

/// recurrsive MultiPointer + length
pub fn MultiSlice(comptime Element: type) type {
    return struct {
        const Slice = @This();
        const Pointer = MultiPointer(Element);
        ptr: Pointer,
        len: usize,

        fn sub(slice: Slice, comptime field: anytype) switch (@typeInfo(Element)) {
            .@"struct" => MultiSlice(field.type),
            .array => |a| MultiSlice(a.child),
            .vector => |v| MultiSlice(v.child),
            else => unreachable,
        } {
            return switch (@typeInfo(Element)) {
                .@"struct" => _: {
                    assert(@TypeOf(field) == std.builtin.Type.StructField);
                    break :_ .{
                        .ptr = @field(slice.ptr, field.name),
                        .len = slice.len,
                    };
                },
                .array, .vector => .{
                    .ptr = slice.ptr[field],
                    .len = slice.len,
                },
                else => unreachable,
            };
        }

        test sub {
            const slice: Slice = undefined;
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        _ = slice.sub(field);
                    }
                },
                .array => |a| {
                    inline for (0..a.len) |i| {
                        _ = slice.sub(i);
                    }
                },
                .vector => |v| {
                    inline for (0..v.len) |i| {
                        _ = slice.sub(i);
                    }
                },
                else => { //end case
                    assert(@typeInfo(Pointer).pointer.size == .many);
                },
            }
        }
    };
}

test MultiSlice {
    _ = MultiSlice(struct { a: bool, b: @Vector(3, u16), c: [4]f32 });
}

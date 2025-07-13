const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

inline fn len(T: type) usize {
    return switch (@typeInfo(T)) {
        .array => |a| a.len,
        .vector => |v| v.len,
        else => unreachable,
    };
}

fn child(T: type) type {
    return switch (@typeInfo(T)) {
        .array => |a| a.child,
        .vector => |v| v.child,
        else => unreachable,
    };
}

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
        .array, .vector => [len(Element)]MultiPointer(child(Element)),
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
            .array, .vector => MultiSlice(child(Element)),
            else => unreachable,
        } {
            return switch (@typeInfo(Element)) {
                .@"struct" => .{
                    .ptr = @field(slice.ptr, field.name),
                    .len = slice.len,
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
                .array, .vector => {
                    inline for (0..len(Element)) |i| {
                        _ = slice.sub(i);
                    }
                },
                else => {
                    assert(Pointer == [*]Element);
                },
            }
        }

        pub fn deinit(slice: Slice, arena: Allocator) void {
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        slice.sub(field).deinit(arena);
                    }
                },
                .array, .vector => {
                    inline for (0..len(Element)) |i| {
                        slice.sub(i).deinit(arena);
                    }
                },
                else => {
                    arena.free(slice.ptr[0..slice.len]);
                },
            }
        }

        pub fn init(n: usize, arena: Allocator) !Slice {
            var slice: Slice = undefined;
            slice.len = n;
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields, 0..) |field, i| {
                        @field(slice.ptr, field.name) = (MultiSlice(field.type).init(n, arena) catch |err| {
                            inline for (0..i) |j| {
                                slice.sub(s.fields[j]).deinit(arena);
                            }
                            return err;
                        }).ptr;
                    }
                },
                .array, .vector => {
                    inline for (0..len(Element)) |i| {
                        slice.ptr[i] = (MultiSlice(child(Element)).init(n, arena) catch |err| {
                            inline for (0..i) |j| {
                                slice.sub(j).deinit(arena);
                            }
                            return err;
                        }).ptr;
                    }
                },
                else => {
                    slice.ptr = (try arena.alloc(Element, n)).ptr;
                },
            }
            return slice;
        }

        test init {
            const ally = std.testing.allocator;
            const a = try Slice.init(10, ally);
            defer a.deinit(ally);
            //? do stuff with a
        }
    };
}

test MultiSlice {
    const A = struct { a: bool, b: @Vector(3, u16), c: [4]f32 };
    _ = MultiSlice(A);
    _ = MultiSlice([5]A);
}

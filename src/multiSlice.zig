const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const simd = @import("simd.zig");
const util = @import("util.zig");

/// recurrsive MultiPointer without functions
fn MultiPointerType(Element: type) type {
    const info_E = @typeInfo(Element);
    return switch (info_E) {
        .@"struct" => |s| _: {
            var fields_MP: [s.fields.len]std.builtin.Type.StructField = undefined;
            for (&fields_MP, s.fields) |*field_MP, field_E| {
                field_MP.* = .{
                    .alignment = @alignOf(field_E.type),
                    .default_value_ptr = null,
                    .is_comptime = field_E.is_comptime,
                    .name = field_E.name,
                    .type = MultiPointerType(field_E.type),
                };
            }
            break :_ @Type(std.builtin.Type{ .@"struct" = .{
                .layout = .auto,
                .fields = &fields_MP,
                .decls = &[_]std.builtin.Type.Declaration{},
                .is_tuple = false,
            } });
        },
        .array, .vector => [util.len(Element)]MultiPointerType(util.Child(Element)),
        else => [*]Element,
    };
}

test MultiPointerType {
    const A = struct { a: bool, b: @Vector(3, u16), c: [4]f32 };
    const B = MultiPointerType(A);
    const b: B = undefined;
    try expect(@TypeOf(b.a) == [*]bool);
    try expect(@TypeOf(b.b) == [3][*]u16);
    try expect(@TypeOf(b.c) == [4][*]f32);
}

/// recurrsive MultiPointer + length
pub fn Type(comptime Element: type) type {
    return struct {
        const MultiSlice = @This();
        const MultiPointer = MultiPointerType(Element);
        ptr: MultiPointer,
        len: usize,

        fn sub(slice: MultiSlice, comptime field: anytype) switch (@typeInfo(Element)) {
            .@"struct" => Type(field.type),
            .array, .vector => Type(util.Child(Element)),
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
            const slice: MultiSlice = undefined;
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        _ = slice.sub(field);
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |i| {
                        _ = slice.sub(i);
                    }
                },
                else => {
                    assert(MultiPointer == [*]Element);
                },
            }
        }

        pub fn deinit(slice: MultiSlice, arena: Allocator) void {
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        slice.sub(field).deinit(arena);
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |i| {
                        slice.sub(i).deinit(arena);
                    }
                },
                else => {
                    arena.free(slice.ptr[0..slice.len]);
                },
            }
        }

        pub fn init(n: usize, arena: Allocator) !MultiSlice {
            var slice: MultiSlice = undefined;
            slice.len = n;
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields, 0..) |field, i| {
                        @field(slice.ptr, field.name) = (Type(field.type).init(n, arena) catch |err| {
                            inline for (0..i) |j| {
                                slice.sub(s.fields[j]).deinit(arena);
                            }
                            return err;
                        }).ptr;
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |i| {
                        slice.ptr[i] = (Type(util.Child(Element)).init(n, arena) catch |err| {
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
            const a = try MultiSlice.init(10, ally);
            defer a.deinit(ally);
        }

        pub inline fn stackInit(comptime n: usize) MultiSlice {
            var slice: MultiSlice = undefined;
            slice.len = n;
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        @field(slice.ptr, field.name) = Type(field.type).stackInit(n).ptr;
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |i| {
                        slice.ptr[i] = Type(util.Child(Element)).stackInit(n).ptr;
                    }
                },
                else => {
                    slice.ptr = util.stackAlloc(Element, n).ptr;
                },
            }
            return slice;
        }

        test stackInit {
            _ = MultiSlice.stackInit(10);
        }

        pub fn set(slice: MultiSlice, i: usize, element: Element) void {
            assert(i < slice.len); //out of bounds
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        slice.sub(field).set(i, @field(element, field.name));
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |j| {
                        slice.sub(j).set(i, element[j]);
                    }
                },
                else => {
                    slice.ptr[i] = element;
                },
            }
        }

        test set {
            const ally = std.testing.allocator;
            const a = try MultiSlice.init(4, ally);
            defer a.deinit(ally);
            const b: Element = undefined;
            a.set(0, b);
            a.set(1, b);
            a.set(2, b);
            a.set(3, b);
        }

        pub fn setN(slice: MultiSlice, simd_len: comptime_int, i: usize, simd_element: simd.Vector(simd_len, Element)) void {
            assert(i + simd_len <= slice.len); //out of bounds
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        slice.sub(field).setN(simd_len, i, @field(simd_element, field.name));
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |j| {
                        slice.sub(j).setN(simd_len, i, simd_element[j]);
                    }
                },
                else => {
                    slice.ptr[i..][0..simd_len].* = simd_element;
                },
            }
        }

        test setN {
            const ally = std.testing.allocator;
            const a = try MultiSlice.init(4, ally);
            defer a.deinit(ally);
            const b: Element = undefined;
            a.setN(1, 0, simd.splat(1, b));
            a.setN(1, 2, simd.splat(1, b));
            a.setN(3, 0, simd.splat(3, b));
            a.setN(3, 1, simd.splat(3, b));
        }

        pub fn at(slice: MultiSlice, i: usize) Element {
            assert(i < slice.len); //out of bounds
            var element: Element = undefined;
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        @field(element, field.name) = slice.sub(field).at(i);
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |j| {
                        element[j] = slice.sub(j).at(i);
                    }
                },
                else => {
                    element = slice.ptr[i];
                },
            }
            return element;
        }

        test at {
            const ally = std.testing.allocator;
            const a = try MultiSlice.init(4, ally);
            defer a.deinit(ally);
            _ = a.at(0);
            _ = a.at(1);
            _ = a.at(2);
            _ = a.at(3);
        }

        pub fn atN(slice: MultiSlice, simd_len: comptime_int, i: usize) simd.Vector(simd_len, Element) {
            assert(i + simd_len <= slice.len); //out of bounds
            var simd_element: simd.Vector(simd_len, Element) = undefined;
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        @field(simd_element, field.name) = slice.sub(field).atN(simd_len, i);
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |j| {
                        simd_element[j] = slice.sub(j).atN(simd_len, i);
                    }
                },
                else => {
                    simd_element = slice.ptr[i..][0..simd_len].*;
                },
            }
            return simd_element;
        }

        test atN {
            const ally = std.testing.allocator;
            const a = try MultiSlice.init(4, ally);
            defer a.deinit(ally);
            _ = a.atN(1, 0);
            _ = a.atN(1, 2);
            _ = a.atN(3, 0);
            _ = a.atN(3, 1);
        }

        pub fn fill(slice: MultiSlice, from: usize, to: usize, element: Element) void {
            assert(from <= to);
            assert(to <= slice.len); //out of bounds
            if (from == to) return;
            switch (@typeInfo(Element)) {
                .@"struct" => |s| {
                    inline for (s.fields) |field| {
                        slice.sub(field).fill(from, to, @field(element, field.name));
                    }
                },
                .array, .vector => {
                    inline for (0..util.len(Element)) |j| {
                        slice.sub(j).fill(from, to, element[j]);
                    }
                },
                else => {
                    @memset(slice.ptr[from..to], element);
                },
            }
        }

        test fill {
            const ally = std.testing.allocator;
            const a = try MultiSlice.init(4, ally);
            defer a.deinit(ally);
            const b: Element = undefined;
            a.fill(0, 0, b);
            a.fill(0, 4, b);
            a.fill(1, 2, b);
            a.fill(4, 4, b);
        }
    };
}

test Type {
    const A = struct {
        a: bool,
        b: @Vector(3, u16),
        c: [4]f32,
    };
    _ = Type(A);
    _ = Type([5]A);
}

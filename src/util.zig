const std = @import("std");
const testing = std.testing;
const expect = testing.expect;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const simd = @import("simd.zig");

/// return the non pointer type
pub fn Deref(A: type) type {
    const info = @typeInfo(A);
    return if (info == .pointer) info.pointer.child else A;
}

test {
    try expect(Deref(usize) == usize);
    try expect(Deref(*usize) == usize);
}

/// ReturnType of the function `f` given arguments of type `Args`.
pub fn ReturnType(comptime f: anytype, comptime Args: type) type {
    const args: Args = undefined;
    return @TypeOf(@call(.auto, f, args));
}

/// ErrorSet of the type `T`.
/// `null` if T contains no error.
pub fn ErrorSet(T: type) ?type {
    return switch (@typeInfo(T)) {
        .error_union => |error_union| error_union.error_set,
        else => null,
    };
}

/// stack allocation
inline fn stackAlloc(Element: type, comptime length: usize) []Element {
    var mem: [length]Element = undefined;
    return mem[0..];
}

inline fn nextDepth(comptime recurrsion_depth: ?usize) ?usize {
    comptime return if (recurrsion_depth) |r| r - 1 else null;
}

/// recurrsive MultiPointer without functions
fn MultiPointer(Element: type, comptime recurrsion_depth: ?usize) type {
    if (recurrsion_depth == 0) return [*]Element;
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
                    .type = MultiPointer(field_E.type, nextDepth(recurrsion_depth)),
                };
            }
            break :_ @Type(std.builtin.Type{ .@"struct" = .{
                .layout = .@"packed",
                .fields = &fields_MP,
                .decls = &[_]std.builtin.Type.Declaration{},
                .is_tuple = false,
            } });
        },
        else => [*]Element,
    };
}

test {
    const A = struct { a: bool, b: u13, c: f32 };
    const As = MultiPointer(A, null);
    const a: As = undefined;
    try expect(@TypeOf(a.a) == [*]bool);
    try expect(@TypeOf(a.b) == [*]u13);
    try expect(@TypeOf(a.c) == [*]f32);
}

test {
    const A = struct { a: @Vector(3, bool), b: @Vector(3, u13), c: @Vector(3, f32) };
    const As = MultiPointer(A, null);
    const a: As = undefined;
    try expect(@TypeOf(a.a) == [*]@Vector(3, bool));
    try expect(@TypeOf(a.b) == [*]@Vector(3, u13));
    try expect(@TypeOf(a.c) == [*]@Vector(3, f32));
}

/// recurrsive MultiPointer + length
pub fn MultiSlice(comptime Element: type, comptime recurrsion_depth: ?usize) type {
    return packed struct {
        const Slice = @This();
        const Pointer = MultiPointer(Element, recurrsion_depth);
        ptr: Pointer,
        len: usize,

        fn sub(slice: Slice, field: std.builtin.Type.StructField) MultiSlice(field.type, nextDepth(recurrsion_depth)) {
            return MultiSlice(field.type, nextDepth(recurrsion_depth)){
                .ptr = @field(slice.ptr, field.name),
                .len = slice.len,
            };
        }

        pub fn init(n: usize, arena: Allocator) !Slice {
            var slice: Slice = undefined;
            slice.len = n;
            slice.ptr = switch (@typeInfo(Pointer)) {
                .@"struct" => _: {
                    const s = @typeInfo(Element).@"struct";
                    var ptr: Pointer = undefined;
                    inline for (s.fields, 0..) |field, i| {
                        @field(ptr, field.name) = (MultiSlice(field.type, nextDepth(recurrsion_depth)).init(n, arena) catch |err| {
                            inline for (0..i) |j| {
                                (Slice{ .len = n, .ptr = ptr }).sub(s.fields[j]).deinit(arena);
                            }
                            return err;
                        }).ptr;
                    }
                    break :_ ptr;
                },
                .pointer => |p| (try arena.alloc(p.child, n)).ptr,
                else => unreachable,
            };
            return slice;
        }

        ///stack allocation
        pub inline fn stackInit(comptime n: usize) Slice {
            var slice: Slice = undefined;
            slice.len = n;
            slice.ptr = switch (@typeInfo(Pointer)) {
                .@"struct" => _: {
                    const s = @typeInfo(Element).@"struct";
                    var ptr: Pointer = undefined;
                    inline for (s.fields) |field| {
                        @field(ptr, field.name) = MultiSlice(field.type, nextDepth(recurrsion_depth)).stackInit(n).ptr;
                    }
                    break :_ ptr;
                },
                .pointer => |p| stackAlloc(p.child, n).ptr,
                else => unreachable,
            };
            return slice;
        }

        pub fn deinit(slice: Slice, allocator: Allocator) void {
            switch (@typeInfo(Pointer)) {
                .@"struct" => {
                    const s = @typeInfo(Element).@"struct";
                    inline for (s.fields) |field| {
                        slice.sub(field).deinit(allocator);
                    }
                },
                .pointer => allocator.free(slice.ptr[0..slice.len]),
                else => unreachable,
            }
        }

        // ensures `.len >= n` but invalidates any content
        pub fn reinit(slice: *Slice, n: usize, allocator: Allocator) !void {
            if (n <= slice.len) return;
            const h = try Slice.init(n, allocator);
            slice.deinit(allocator);
            slice.* = h;
        }

        pub fn set(slice: Slice, i: usize, element: Element) void {
            slice.setN(1, i, element);
        }

        pub fn setN(slice: Slice, simd_len: comptime_int, i: usize, simd_element: simd.Vector(simd_len, Element)) void {
            if (simd_len > 1) assert(recurrsion_depth == null); // only supported for full SOA
            assert(i + simd_len <= slice.len);
            switch (@typeInfo(Pointer)) {
                .@"struct" => {
                    const s = @typeInfo(Element).@"struct";
                    inline for (s.fields) |field| {
                        slice.sub(field).setN(simd_len, i, @field(simd_element, field.name));
                    }
                },
                .pointer => {
                    if (simd_len > 1) {
                        slice.ptr[i..][0..simd_len].* = simd_element;
                    } else {
                        slice.ptr[i] = simd_element;
                    }
                },
                else => unreachable,
            }
        }

        pub fn at(slice: Slice, i: usize) Element {
            return slice.atN(1, i);
        }

        pub fn atN(slice: Slice, simd_len: comptime_int, i: usize) Element {
            if (simd_len > 1) assert(recurrsion_depth == null); // only supported for full SOA
            assert(i + simd_len <= slice.len);
            return switch (@typeInfo(Pointer)) {
                .@"struct" => _: {
                    const s = @typeInfo(Element).@"struct";
                    var element: Element = undefined;
                    inline for (s.fields) |field| {
                        @field(element, field.name) = slice.sub(field).at(i);
                    }
                    break :_ element;
                },
                .pointer => _: {
                    if (simd_len > 1) {
                        break :_ slice.ptr[i..][0..simd_len].*;
                    } else {
                        break :_ slice.ptr[i];
                    }
                },
                else => unreachable,
            };
        }

        pub fn fill(slice: Slice, from: usize, to: usize, element: Element) void {
            assert(simd.length(Element) == 1);
            switch (@typeInfo(Pointer)) {
                .@"struct" => {
                    const s = @typeInfo(Element).@"struct";
                    inline for (s.fields) |field| {
                        slice.sub(field).fill(from, to, @field(element, field.name));
                    }
                },
                .pointer => @memset(slice.ptr[from..to], element),
                else => unreachable,
            }
        }
    };
}

test "MultiSlice" {
    const ally = testing.allocator;
    const A = struct { a: bool, b: u13, c: f32 };
    const C = simd.Vector(3, A);
    const SA = MultiSlice(A, null);

    //init/deinit
    const sa = try SA.init(5, ally);
    defer sa.deinit(ally);
    try expect(@TypeOf(sa.ptr.a) == [*]bool);
    try expect(@TypeOf(sa.ptr.b) == [*]u13);
    try expect(@TypeOf(sa.ptr.c) == [*]f32);
    try expect(sa.len == 5);
    //try expect(SC.simd_size == 3);

    //at/set
    const c = C{
        .a = .{ true, false, false },
        .b = .{ 3, 4, 5 },
        .c = .{ -0.5, 7.25, -3.75 },
    };
    sa.setN(3, 0, c);
    sa.setN(3, 2, c);
    try expect(sa.at(0).a == true);
    try expect(sa.at(1).a == false);
    try expect(sa.at(2).a == true);
    try expect(sa.at(3).a == false);
    try expect(sa.at(4).a == false);

    try expect(sa.at(0).b == 3);
    try expect(sa.at(1).b == 4);
    try expect(sa.at(2).b == 3);
    try expect(sa.at(3).b == 4);
    try expect(sa.at(4).b == 5);

    try expect(sa.at(0).c == -0.5);
    try expect(sa.at(1).c == 7.25);
    try expect(sa.at(2).c == -0.5);
    try expect(sa.at(3).c == 7.25);
    try expect(sa.at(4).c == -3.75);
}

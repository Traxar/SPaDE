const std = @import("std");
const testing = std.testing;
const expect = testing.expect;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const simd = @import("simd.zig");

/// return the non pointer type
pub fn Deref(A: type) type {
    const info = @typeInfo(A);
    return if (info == .Pointer) info.Pointer.child else A;
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
        .ErrorUnion => |error_union| error_union.error_set,
        else => null,
    };
}

/// stack allocation
inline fn stackAlloc(Element: type, comptime length: usize) []Element {
    var mem: [length]Element = undefined;
    return mem[0..];
}

/// recurrsive MultiPointer without functions
fn MultiPointer(Element: type) type {
    const info_E = @typeInfo(Element);
    return switch (info_E) {
        .Struct => |s| _: {
            var fields_MP: [s.fields.len]std.builtin.Type.StructField = undefined;
            for (&fields_MP, s.fields) |*field_MP, field_E| {
                field_MP.* = .{
                    .alignment = 0,
                    .default_value = null,
                    .is_comptime = field_E.is_comptime,
                    .name = field_E.name,
                    .type = MultiPointer(field_E.type),
                };
            }
            break :_ @Type(std.builtin.Type{ .Struct = .{
                .layout = .@"packed",
                .fields = &fields_MP,
                .decls = &[_]std.builtin.Type.Declaration{},
                .is_tuple = false,
            } });
        },
        .Vector => |v| [*]v.child,
        else => [*]Element,
    };
}

test {
    const A = struct { a: bool, b: u13, c: f32 };
    const C = MultiPointer(A);
    const c: C = undefined;
    try expect(@TypeOf(c.a) == [*]bool);
    try expect(@TypeOf(c.b) == [*]u13);
    try expect(@TypeOf(c.c) == [*]f32);
}

/// recurrsive MultiPointer + length
pub fn MultiSlice(comptime Element: type) type {
    return packed struct {
        const Slice = @This();
        ptr: MultiPointer(Element),
        len: usize,

        const simd_size = simd.length(Element);

        fn sub(slice: Slice, field: std.builtin.Type.StructField) MultiSlice(field.type) {
            return MultiSlice(field.type){
                .ptr = @field(slice.ptr, field.name),
                .len = slice.len,
            };
        }

        // TODO: #4: fuse into single alloc
        pub fn init(n: usize, allocator: Allocator) !Slice {
            var slice: Slice = undefined;
            slice.len = n;
            slice.ptr = switch (@typeInfo(Element)) {
                .Struct => |s| _: {
                    var res: MultiPointer(Element) = undefined;
                    inline for (s.fields) |field| {
                        @field(res, field.name) = (try MultiSlice(field.type).init(n, allocator)).ptr;
                    }
                    break :_ res;
                },
                .Vector => |v| (try allocator.alloc(v.child, n)).ptr,
                else => (try allocator.alloc(Element, n)).ptr,
            };
            return slice;
        }

        ///stack allocation
        pub inline fn stackInit(comptime n: usize) Slice {
            var slice: Slice = undefined;
            slice.len = n;
            slice.ptr = switch (@typeInfo(Element)) {
                .Struct => |s| _: {
                    var res: MultiPointer(Element) = undefined;
                    inline for (s.fields) |field| {
                        @field(res, field.name) = MultiSlice(field.type).stackInit(n).ptr;
                    }
                    break :_ res;
                },
                .Vector => |v| stackAlloc(v.child, n).ptr,
                else => stackAlloc(Element, n).ptr,
            };
            return slice;
        }

        pub fn deinit(slice: Slice, allocator: Allocator) void {
            switch (@typeInfo(Element)) {
                .Struct => |s| {
                    inline for (s.fields) |field| {
                        slice.sub(field).deinit(allocator);
                    }
                },
                else => allocator.free(slice.ptr[0..slice.len]),
            }
        }

        // ensures capacity but invalidates any content
        pub fn ensureCapacity(slice: *Slice, n: usize, allocator: Allocator) !void {
            if (n <= slice.len) return;
            const h = try Slice.init(n, allocator);
            slice.deinit(allocator);
            slice.* = h;
        }

        pub fn set(slice: Slice, i: usize, element: Element) void {
            assert(i + simd_size <= slice.len);
            switch (@typeInfo(Element)) {
                .Struct => |s| {
                    inline for (s.fields) |field| {
                        slice.sub(field).set(i, @field(element, field.name));
                    }
                },
                .Vector => slice.ptr[i..][0..simd_size].* = element,
                else => slice.ptr[i] = element,
            }
        }

        pub fn at(slice: Slice, i: usize) Element {
            assert(i + simd_size <= slice.len);
            return switch (@typeInfo(Element)) {
                .Struct => |s| _: {
                    var element: Element = undefined;
                    inline for (s.fields) |field| {
                        @field(element, field.name) = slice.sub(field).at(i);
                    }
                    break :_ element;
                },
                .Vector => slice.ptr[i..][0..simd_size].*,
                else => slice.ptr[i],
            };
        }

        pub fn fill(slice: Slice, from: usize, to: usize, element: Element) void {
            assert(simd.length(Element) == 1);
            switch (@typeInfo(Element)) {
                .Struct => |s| {
                    inline for (s.fields) |field| {
                        slice.sub(field).fill(from, to, @field(element, field.name));
                    }
                },
                else => @memset(slice.ptr[from..to], @bitCast(element)),
            }
        }
    };
}

test "MultiSlice" {
    const ally = testing.allocator;
    const A = struct { a: bool, b: u13, c: f32 };
    const C = simd.Vector(3, A);
    const SA = MultiSlice(A);
    const SC = MultiSlice(C);

    //init/deinit
    const sc = try SC.init(5, ally);
    defer sc.deinit(ally);
    try expect(@TypeOf(sc.ptr.a) == [*]bool);
    try expect(@TypeOf(sc.ptr.b) == [*]u13);
    try expect(@TypeOf(sc.ptr.c) == [*]f32);
    try expect(sc.len == 5);
    try expect(SC.simd_size == 3);

    //at/set
    const c = C{
        .a = .{ true, false, false },
        .b = .{ 3, 4, 5 },
        .c = .{ -0.5, 7.25, -3.75 },
    };
    sc.set(0, c);
    sc.set(2, c);
    const sa: SA = @bitCast(sc);
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

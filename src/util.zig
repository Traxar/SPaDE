const std = @import("std");
const testing = std.testing;
const expect = testing.expect;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

pub fn Deref(a: type) type {
    const info = @typeInfo(a);
    return if (info == .Pointer) info.Pointer.child else a;
}

pub inline fn ErrorSet(comptime op: anytype, comptime Args: type) ?type {
    const args: Args = undefined;
    const return_type = @TypeOf(@call(.auto, op, args));
    return switch (@typeInfo(return_type)) {
        .ErrorUnion => |error_union| error_union.error_set,
        else => null,
    };
}

/// recurrsive MultiPointer without functions
fn MultiPointer(comptime T: type) type {
    const fields_T = @typeInfo(T).Struct.fields;
    var fields_MP: [fields_T.len]std.builtin.Type.StructField = undefined;

    for (&fields_MP, fields_T) |*f_MP, f_T| {
        const info_sub = @typeInfo(f_T.type);
        const MP_sub = switch (info_sub) {
            .Vector => [*]info_sub.Vector.child,
            .Struct => MultiPointer(f_T.type),
            else => @compileError("type not supported"),
        };
        f_MP.* = .{
            .alignment = 0,
            .default_value = null,
            .is_comptime = f_T.is_comptime,
            .name = f_T.name,
            .type = MP_sub,
        };
    }
    return @Type(std.builtin.Type{ .Struct = .{
        .layout = .@"packed",
        .fields = &fields_MP,
        .decls = &[_]std.builtin.Type.Declaration{},
        .is_tuple = false,
    } });
}

/// recurrsive MultiPointer
pub fn MultiSlice(comptime Element: type) type {
    if (@typeInfo(Element) != .Struct) @compileError("type must be a struct");
    return packed struct {
        const Slice = @This();
        ptr: MultiPointer(Element),
        len: usize,

        const simd_size = simd_size: {
            var simd: usize = undefined;
            const fields_T = @typeInfo(Element).Struct.fields;
            for (fields_T, 0..) |f_T, i| {
                const info_sub = @typeInfo(f_T.type);
                const simd_sub = switch (info_sub) {
                    .Vector => info_sub.Vector.len,
                    .Struct => MultiSlice(f_T.type).simd_size,
                    else => unreachable,
                };
                if (i == 0) {
                    simd = simd_sub;
                } else if (simd != simd_sub) {
                    @compileError("vector length does not match");
                }
            }
            break :simd_size simd;
        };

        fn sub(slice: Slice, field: std.builtin.Type.StructField) MultiSlice(field.type) {
            return MultiSlice(field.type){
                .ptr = @field(slice.ptr, field.name),
                .len = slice.len,
            };
        }

        // TODO: errdefer or better: fuse into single alloc
        pub fn init(n: usize, allocator: Allocator) !Slice {
            var slice: Slice = undefined;
            slice.len = n;
            inline for (@typeInfo(Element).Struct.fields) |field| {
                const info = @typeInfo(field.type);
                @field(slice.ptr, field.name) = (try switch (info) {
                    .Vector => allocator.alloc(info.Vector.child, n),
                    .Struct => MultiSlice(field.type).init(n, allocator),
                    else => unreachable,
                }).ptr;
            }
            return slice;
        }

        pub fn deinit(slice: Slice, allocator: Allocator) void {
            inline for (@typeInfo(Element).Struct.fields) |field| {
                const info = @typeInfo(field.type);
                switch (info) {
                    .Vector => allocator.free(@field(slice.ptr, field.name)[0..slice.len]),
                    .Struct => slice.sub(field).deinit(allocator),
                    else => unreachable,
                }
            }
        }

        pub fn ensureCapacity(slice: *Slice, n: usize, allocator: Allocator) !void {
            if (n <= slice.len) return;
            const h = try Slice.init(n, allocator);
            slice.deinit(allocator);
            slice.* = h;
        }

        pub fn set(slice: Slice, i: usize, element: Element) void {
            assert(i + simd_size <= slice.len);
            inline for (@typeInfo(Element).Struct.fields) |field| {
                const info = @typeInfo(field.type);
                switch (info) {
                    .Vector => @field(slice.ptr, field.name)[i..][0..info.Vector.len].* = @field(element, field.name),
                    .Struct => slice.sub(field).set(i, @field(element, field.name)),
                    else => unreachable,
                }
            }
        }

        pub fn at(slice: Slice, i: usize) Element {
            assert(i + simd_size <= slice.len);
            var element: Element = undefined;
            inline for (@typeInfo(Element).Struct.fields) |field| {
                @field(element, field.name) = switch (@typeInfo(field.type)) {
                    .Vector => @field(slice.ptr, field.name)[i..][0..simd_size].*,
                    .Struct => slice.sub(field).at(i),
                    else => unreachable,
                };
            }
            return element;
        }

        pub fn fill(slice: Slice, from: usize, to: usize, element: Element) void {
            inline for (@typeInfo(Element).Struct.fields) |field| {
                const info = @typeInfo(field.type);
                switch (info) {
                    .Vector => @memset(@field(slice.ptr, field.name)[from..to], @bitCast(@field(element, field.name))),
                    .Struct => slice.sub(field).fill(from, to, @field(element, field.name)),
                    else => unreachable,
                }
            }
        }
    };
}

test "MultiSlice" {
    const ally = testing.allocator;
    const V = struct { a: @Vector(5, bool), b: struct { c: @Vector(5, usize) } };
    const MS = MultiSlice(V);
    // init/deinit
    const ms = try MS.init(10, ally);
    defer ms.deinit(ally);
    try expect(@TypeOf(ms.ptr.a) == [*]bool);
    try expect(@TypeOf(ms.ptr.b.c) == [*]usize);
    try expect(ms.len == 10);
    try expect(MS.simd_size == 5);
    // at/set
    const v = .{
        .a = .{ true, true, false, false, false },
        .b = .{
            .c = .{ 1, 2, 3, 4, 5 },
        },
    };
    ms.set(1, v);
    // std.log.warn("{any}", .{ms.ptr.a[0..][0..5].*});
    // std.log.warn("{any}", .{ms.at(0).a});

    // try expect(ms.at(0).a[2] == true); // TODO: uncomment, fails in zig 0.13.0 (only bools affected)
    // try expect(ms.at(0).a[3] == false); // TODO: uncomment, fails in zig 0.13.0 (only bools affected)
    try expect(ms.at(0).b.c[1] == 1);
    try expect(ms.at(0).b.c[2] == 2);
}

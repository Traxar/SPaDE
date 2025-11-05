const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const MultiSlice = @import("multiSlice.zig").Type;
const Dims = @import("dims.zig").Type;
const Layout = @import("layout.zig").Type;
const Coords = @import("coords.zig").Type;
const Arg = @import("args.zig").Type;
const simd = @import("simd.zig");

/// Returns `true` if `T` is a sparse tensor.
pub inline fn is(T: type) bool {
    comptime {
        if (@typeInfo(T) != .@"struct") return false;
        if (!@hasDecl(T, "Element") or @TypeOf(T.Element) != type) return false;
        if (!@hasDecl(T, "dims_sparse") or @TypeOf(T.dims_sparse) != Dims) return false;
        if (!@hasDecl(T, "dims_dense") or @TypeOf(T.dims_dense) != Dims) return false;
        if (!@hasDecl(T, "zero") or @TypeOf(T.zero) != T.Element) return false;
        return T == Type(T.Element, T.dims_sparse, T.dims_dense, T.zero);
    }
}

pub fn Type(_Index: type, _Element: type, comptime dims_sparse_: Dims, comptime dims_dense_: Dims, comptime zero_: _Element) type {
    return struct {
        const Sparse = @This();
        pub const Index = _Index;
        pub const Element = _Element;
        pub const sparse_dims = dims_sparse_;
        pub const dense_dims = dims_dense_;
        pub const zero = zero_;

        const Entry = struct { ind: usize, val: Element };
        const Indices = MultiSlice(usize);
        const Entries = MultiSlice(Entry);

        min_dense_index: usize, //first dense index with entries
        max_dense_index: usize, //1 + last dense index with entries
        inds: Indices,
        ents: Entries,
        dense_layout: Layout(Index, dense_dims),
        sparse_layout: Layout(Index, sparse_dims),

        /// Returns a newly allocated tensor with size `size`.
        /// - `size` is a multiindex. All entries at an index not in `dims` will be ignored.
        /// - all values are set to 'zero'
        /// - there is capacity for 'nonzeros' nonzeros
        /// - Free the result by using `deinit`.
        ///
        /// (ex.: `Tensor(f32).Sparse(&.{0},&.{1},0).init(&.{200, 300})` tries to allocate a 200x300 matrix)
        pub fn init(size_: []const Index, nonzeros: usize, arena: Allocator) !Sparse {
            const dense_layout = Layout(Index, dense_dims).from(size_);
            const inds = try Indices.init(dense_layout.n() + 1, arena);
            errdefer inds.deinit(arena);
            const sparse_layout = Layout(Index, sparse_dims).from(size_);
            const ents = try Entries.init(nonzeros, arena);
            errdefer ents.deinit(arena);
            return .{
                .min_dense_index = 0,
                .max_dense_index = 0,
                .dense_layout = dense_layout,
                .sparse_layout = sparse_layout,
                .inds = inds,
                .ents = ents,
            };
        }

        /// Sets the size of tensor `tensor` to the new size `size`, while reusing the already allocated memory if possible.
        /// - `size` is a multiindex. All entries at an index not in `dims` will be ignored.
        /// - invalidates data
        pub fn reinit(tensor: *Sparse, size_: []const usize, nonzeros: usize, arena: Allocator) !void {
            const dense_layout = Layout(dense_dims).from(size_);
            try tensor.inds.reinit(dense_layout.n() + 1, arena);
            tensor.dense_layout = dense_layout;
            tensor.min_dense_index = 0;
            tensor.max_dense_index = 0;
            try tensor.ents.reinit(nonzeros, arena);
            tensor.sparse_layout = Layout(sparse_dims).from(size_);
        }

        /// Frees memory allocated by tensor `tensor`.
        pub fn deinit(tensor: Sparse, arena: Allocator) void {
            tensor.inds.deinit(arena);
            tensor.ents.deinit(arena);
        }

        /// Returns the size of tensor `tensor` in dimension `d`
        pub fn size(tensor: Sparse, comptime d: usize) usize {
            return tensor.sparse_layout.size.at(d) + tensor.dense_layout.size.at(d);
        }

        pub fn setCursor(tensor: Sparse, index: usize) void {
            assert(tensor.min_dense_index == tensor.max_dense_index); //can only set cursor if tensor does not contain any entries
            assert(index < tensor.ents.len);
            tensor.inds.set(tensor.min_dense_index, index);
        }

        fn eq(a: Element, b: Element) bool {
            return std.meta.eql(a, b);
        }

        /// Returns
        /// - {value, index} if entry exists
        /// - null if no entry at `dense_index` + `sparse_index`
        fn search(tensor: Sparse, dense_index: usize, sparse_index: usize) ?struct { val: Element, i: usize } {
            if (dense_index < tensor.min_dense_index) return null;
            if (dense_index >= tensor.max_dense_index) return null;
            var min = tensor.inds.at(dense_index);
            var max = tensor.inds.at(dense_index + 1);
            max = @min(max, min + sparse_index + 1);
            // min = @max(min, max - ...) //TODO
            if (max == min) return null;
            while (max - min > 1) {
                const mid = @divFloor(min + max, 2);
                if (tensor.ents.at(mid).ind < sparse_index)
                    max = mid
                else
                    min = mid;
            }
            const ent = tensor.ents.at(min);
            if (ent.ind != sparse_index) return null;
            return .{ .val = ent.val, .i = min };
        }

        pub fn at(tensor: Sparse, coords: []const Index) Element {
            const dense_index = tensor.dense_layout.index(coords);
            const sparse_index = tensor.sparse_layout.index(coords);
            const search_result = tensor.search(dense_index, sparse_index) orelse return zero;
            return search_result.val;
        }

        /// Sets tensor `tensor` at coordinates `coords` to the new value `value`.
        /// - `coord` is a multiindex. All entries at an index not in `dims` will be ignored.
        pub fn set(tensor: *Sparse, coords: []const Index, value: Element) !void {
            const dense_index = tensor.dense_layout.index(coords);
            const sparse_index = tensor.sparse_layout.index(coords);
            const search_result = tensor.search(dense_index, sparse_index);
            const min_index = tensor.inds.at(tensor.min_dense_index);
            const max_index = tensor.inds.at(tensor.max_dense_index);
            const first = dense_index < tensor.min_dense_index or (dense_index == tensor.min_dense_index and sparse_index <= tensor.ents.at(min_index).ind);
            const last = dense_index >= tensor.max_dense_index or (dense_index + 1 == tensor.max_dense_index and sparse_index >= tensor.ents.at(max_index - 1).ind);
            if (search_result) |res| { //entry already exists
                if (!(first or last) or !eq(value, zero)) {
                    tensor.ents.set(res.i, .{ .val = value, .ind = sparse_index });
                } else if (last) { // value == zero and last
                    var new_max_index = res.i;
                    while (new_max_index > min_index and eq(tensor.ents.at(new_max_index - 1).val, zero)) {
                        new_max_index -= 1;
                    }
                    while (tensor.max_dense_index > tensor.min_dense_index and tensor.inds.at(tensor.max_dense_index - 1) >= new_max_index) {
                        tensor.max_dense_index -= 1;
                    }
                    tensor.inds.set(tensor.max_dense_index, new_max_index);
                } else if (first) { // value == zero and first
                    var new_min_index = res.i + 1;
                    while (new_min_index < max_index and eq(tensor.ents.at(new_min_index).val, zero)) {
                        new_min_index += 1;
                    }
                    while (tensor.min_dense_index < tensor.max_dense_index and tensor.inds.at(tensor.min_dense_index + 1) <= new_min_index) {
                        tensor.min_dense_index += 1;
                    }
                    tensor.inds.set(tensor.min_dense_index, new_min_index);
                }
            } else if (!eq(value, zero)) { //search == null --> new entry
                const empty = tensor.min_dense_index == tensor.max_dense_index;
                if (empty) {
                    @branchHint(.cold);
                    const i = tensor.inds.at(tensor.min_dense_index);
                    tensor.ents.set(i, .{ .val = value, .ind = sparse_index });
                    tensor.inds.set(dense_index, i);
                    tensor.inds.set(dense_index + 1, i + 1);
                    tensor.max_dense_index = dense_index + 1;
                    tensor.min_dense_index = dense_index;
                    return;
                } else if (last) {
                    const i = tensor.inds.at(tensor.max_dense_index);
                    if (i >= tensor.ents.len) return error.OutOfMemory;
                    tensor.ents.set(i, .{ .val = value, .ind = sparse_index });
                    if (tensor.max_dense_index < dense_index)
                        tensor.inds.fill(tensor.max_dense_index + 1, dense_index + 1, i);
                    tensor.inds.set(dense_index + 1, i + 1);
                    tensor.max_dense_index = dense_index + 1;
                } else if (first) {
                    const i = tensor.inds.at(tensor.min_dense_index);
                    if (i == 0) return error.OutOfMemory;
                    tensor.ents.set(i - 1, .{ .val = value, .ind = sparse_index });
                    if (dense_index + 1 < tensor.min_dense_index)
                        tensor.inds.fill(dense_index + 1, tensor.min_dense_index, i);
                    tensor.inds.set(dense_index, i - 1);
                    tensor.min_dense_index = dense_index;
                } else {
                    return error.InsertNotAllowed;
                }
            }
        }
    };
}

test {
    const ally = std.testing.allocator;
    const S = Type(u16, f32, Dims.from(&.{ 1, 2 }), Dims.from(&.{0}), 0);
    const s = try S.init(&.{ 3, 4, 5, 6 }, 7, ally); //unused dimension 3 is ignored
    defer s.deinit(ally);
    try expect(s.inds.len == 3 + 1);
    try expect(s.ents.len == 7);
}

test {
    const ally = std.testing.allocator;
    const S = Type(u16, f32, Dims.from(&.{1}), Dims.from(&.{0}), 0); //sparse rows
    var s = try S.init(&.{ 10, 10 }, 20, ally); //unused dimension 3 is ignored
    defer s.deinit(ally);
    try expect(s.inds.len == 11);
    try expect(s.ents.len == 20);

    try expect(s.at(&.{ 0, 0 }) == 0);
    try expect(s.at(&.{ 0, 9 }) == 0);
    try expect(s.at(&.{ 9, 0 }) == 0);
    try expect(s.at(&.{ 9, 9 }) == 0);

    s.setCursor(0);
    try s.set(&.{ 0, 0 }, 1);
    try s.set(&.{ 0, 9 }, 2);
    try s.set(&.{ 9, 0 }, 3);
    try s.set(&.{ 9, 9 }, 4);
    try expect(s.at(&.{ 0, 0 }) == 1);
    try expect(s.at(&.{ 0, 9 }) == 2);
    try expect(s.at(&.{ 9, 0 }) == 3);
    try expect(s.at(&.{ 9, 9 }) == 4);

    try s.set(&.{ 9, 9 }, 0);
    try expect(s.at(&.{ 0, 0 }) == 1);
    try expect(s.at(&.{ 0, 9 }) == 2);
    try expect(s.at(&.{ 9, 0 }) == 3);
    try expect(s.at(&.{ 9, 9 }) == 0);
    try expect(s.min_dense_index == 0);
    try expect(s.max_dense_index == 10);
    try expect(s.inds.at(0) == 0);
    try expect(s.inds.at(1) == 2);
    try expect(s.inds.at(2) == 2);
    try expect(s.inds.at(9) == 2);
    try expect(s.inds.at(10) == 3);

    try s.set(&.{ 0, 9 }, 0);
    try s.set(&.{ 9, 0 }, 0);
    try expect(s.at(&.{ 0, 0 }) == 1);
    try expect(s.at(&.{ 0, 9 }) == 0);
    try expect(s.at(&.{ 9, 0 }) == 0);
    try expect(s.at(&.{ 9, 9 }) == 0);
    try expect(s.min_dense_index == 0);
    try expect(s.max_dense_index == 1);
    try expect(s.inds.at(0) == 0);
    try expect(s.inds.at(1) == 1);
}

test {
    const ally = std.testing.allocator;
    const S = Type(u16, f32, Dims.from(&.{1}), Dims.from(&.{0}), 0); //sparse rows
    var s = try S.init(&.{ 10, 10 }, 20, ally); //unused dimension 3 is ignored
    defer s.deinit(ally);
    try expect(s.inds.len == 11);
    try expect(s.ents.len == 20);

    try expect(s.at(&.{ 0, 0 }) == 0);
    try expect(s.at(&.{ 0, 9 }) == 0);
    try expect(s.at(&.{ 9, 0 }) == 0);
    try expect(s.at(&.{ 9, 9 }) == 0);

    s.setCursor(s.ents.len - 1);
    try s.set(&.{ 9, 9 }, 4);
    try s.set(&.{ 9, 0 }, 3);
    try s.set(&.{ 0, 9 }, 2);
    try s.set(&.{ 0, 0 }, 1);
    try expect(s.at(&.{ 0, 0 }) == 1);
    try expect(s.at(&.{ 0, 9 }) == 2);
    try expect(s.at(&.{ 9, 0 }) == 3);
    try expect(s.at(&.{ 9, 9 }) == 4);

    try s.set(&.{ 0, 0 }, 0);
    try s.set(&.{ 0, 9 }, 0);
    try expect(s.at(&.{ 0, 0 }) == 0);
    try expect(s.at(&.{ 0, 9 }) == 0);
    try expect(s.at(&.{ 9, 0 }) == 3);
    try expect(s.at(&.{ 9, 9 }) == 4);
    try expect(s.min_dense_index == 9);
    try expect(s.max_dense_index == 10);
    try expect(s.inds.at(9) == 18);
    try expect(s.inds.at(10) == 20);

    try s.set(&.{ 9, 0 }, 0);
    try s.set(&.{ 0, 9 }, 0);
    try expect(s.at(&.{ 0, 0 }) == 0);
    try expect(s.at(&.{ 0, 9 }) == 0);
    try expect(s.at(&.{ 9, 0 }) == 0);
    try expect(s.at(&.{ 9, 9 }) == 4);
    try expect(s.min_dense_index == 9);
    try expect(s.max_dense_index == 10);
    try expect(s.inds.at(9) == 19);
    try expect(s.inds.at(10) == 20);

    try s.set(&.{ 9, 9 }, 0);
    try expect(s.at(&.{ 9, 9 }) == 0);
    try expect(s.min_dense_index == 9);
    try expect(s.max_dense_index == 9);
    try expect(s.inds.at(9) == 19);
}

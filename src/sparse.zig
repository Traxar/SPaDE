const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const util = @import("util.zig");
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

pub fn Type(Element_: type, comptime dims_sparse_: Dims, comptime dims_dense_: Dims, comptime zero_: Element_) type {
    return packed struct {
        const Sparse = @This();
        pub const Element = Element_;
        pub const dims_sparse = dims_sparse_;
        pub const dims_dense = dims_dense_;
        pub const zero = zero_;

        const Entry = struct { ind: usize, val: Element };
        const DataDense = util.MultiSlice(usize, null);
        const DataSparse = util.MultiSlice(Entry, 1);

        start: usize,
        stop: usize,
        inds_dense: DataDense,
        ents_sparse: DataSparse,
        layout_dense: Layout(dims_dense),
        layout_sparse: Layout(dims_sparse),

        /// Returns a newly allocated tensor with size `size`.
        /// - `size` is a multiindex. All entries at an index not in `dims` will be ignored.
        /// - all values are set to 'zero'
        /// - there is capacity for 'nonzeros' nonzeros
        /// - Free the result by using `deinit`.
        ///
        /// (ex.: `Tensor(f32).Sparse(&.{0},&.{1},0).init(&.{200, 300})` tries to allocate a 200x300 matrix)
        pub fn init(size_: []const usize, nonzeros: usize, arena: Allocator) !Sparse {
            const layout_dense = Layout(dims_dense).from(size_);
            const inds_dense = try DataDense.init(layout_dense.n() + 1, arena);
            errdefer inds_dense.deinit(arena);
            const layout_sparse = Layout(dims_sparse).from(size_);
            const ents_sparse = try DataSparse.init(nonzeros, arena);
            errdefer ents_sparse.deinit(arena);
            return .{
                .start = 0,
                .stop = 0,
                .layout_dense = layout_dense,
                .layout_sparse = layout_sparse,
                .inds_dense = inds_dense,
                .ents_sparse = ents_sparse,
            };
        }

        /// Sets the size of tensor `tensor` to the new size `size`, while reusing the already allocated memory if possible.
        /// - `size` is a multiindex. All entries at an index not in `dims` will be ignored.
        /// - invalidates data
        pub fn reinit(tensor: *Sparse, size_: []const usize, nonzeros: usize, arena: Allocator) !void {
            const layout_dense = Layout(dims_dense).from(size_);
            try tensor.inds_dense.reinit(layout_dense.n() + 1, arena);
            tensor.layout_dense = layout_dense;
            tensor.start = 0;
            tensor.stop = 0;
            try tensor.ents_sparse.reinit(nonzeros, arena);
            tensor.layout_sparse = Layout(dims_sparse).from(size_);
        }

        /// Frees memory allocated by tensor `tensor`.
        pub fn deinit(tensor: Sparse, arena: Allocator) void {
            tensor.inds_dense.deinit(arena);
            tensor.ents_sparse.deinit(arena);
        }

        /// Returns the size of tensor `tensor` in dimension `d`
        pub fn size(tensor: Sparse, comptime d: usize) usize {
            return tensor.layout_sparse.size.at(d) + tensor.layout_dense.size.at(d);
        }

        // fn search(tensor: Sparse, coords: []const usize) struct { ind: usize, ex: bool } {
        //     const index_dense = tensor.layout_dense.index(coords);
        //     if (index_dense < tensor.start) return .{.ind = tensor.inds_dense.at(index_dense), .ex = false};// ???
        //     if (tensor.stop <= index_dense) return .{.ind = tensor.inds_dense.at(index_dense+1), .ex = false};

        //     var min = tensor.inds_dense.at(index_dense);
        //     var max = tensor.inds_dense.at(index_dense + 1);
        //     if (min == return
        //     //use sparse size to maybe reduce search

        //     //binary search

        // }

        // /// return index of row, col in matrix
        // /// performs binary search
        // /// O(log(m)), O(1) if dense
        // fn indAt(a: Matrix, row: Index, col: Index) I {
        //     const r = a.val.items(.col)[a.rptr[row]..a.rptr[row + 1]];
        //     if (r.len == 0) {
        //         return .{ .ind = a.rptr[row], .ex = false };
        //     } else {
        //         var min: Index = @intCast(r.len -| (a.cols - col));
        //         var max: Index = @intCast(@min(r.len - 1, col));
        //         while (min < max) {
        //             const pivot = @divFloor(min + max, 2);
        //             if (col <= r[pivot]) {
        //                 max = pivot;
        //             } else {
        //                 min = pivot + 1;
        //             }
        //         }
        //         return .{ .ind = a.rptr[row] + min, .ex = col == r[min] };
        //     }
        // }

    };
}

test {
    const ally = std.testing.allocator;
    const S = Type(f32, Dims.from(&.{ 1, 2 }), Dims.from(&.{0}), 0);
    const s = try S.init(&.{ 3, 4, 5, 6 }, 7, ally); //unused dimension 3 is ignored
    defer s.deinit(ally);
    try expect(s.inds_dense.len == 3 + 1);
    try expect(s.ents_sparse.len == 7);
}

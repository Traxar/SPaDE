const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const MultiSlice = @import("multiSlice.zig").Type;
const Dims = @import("dims.zig").Type;
const Arg = @import("args.zig").Type;
const dense = @import("dense.zig");
const sparse = @import("sparse.zig");
const simd = @import("simd.zig");

/// Returns `true` if `T` is a tensor.
pub inline fn is(T: type) bool {
    comptime {
        return dense.is(T) or sparse.is(T);
    }
}

/// Returns a base type used for:
/// - creating tensor types with elements of type `Element`.
/// - tensor operations with a single scalar output of type `Element`.
pub fn Type(Index: type, Element: type) type {
    if (is(Element)) @compileError("Element must not be a tensor");
    return packed struct {
        /// Returns a dense tensor type with dimensions specified by `dims`:
        /// - `dims.len` specifies the order of the tensor. (ex.: `&.{0, 1}` results in a 2D-tensor aka a matrix)
        /// - the values of a tensor can be accessed via a multiindex which also has the type `[]usize`
        /// - the values of `dims` specify, which entries of the multiindex will be used.
        ///   (ex.: `&.{0, 1}` maps index-`0` and index-`1` onto the matrix, all other/further indices will be ingnored)
        /// - the order of `dims` specifies the memory layout.
        ///   (ex.: `&.{0, 1}` when scanning the allocated memory, index-`0` is scanned first before moving to the next value of index-`1`)
        pub fn Dense(comptime dims: []const usize) type {
            return dense.Type(Index, Element, Dims.from(dims));
        }

        /// Returns a sparse tensor type with dimensions specified by `dims':
        /// - `dims.len` specifies the order of the tensor. (ex.: `&.{0, 1}` results in a 2D-tensor aka a matrix)
        /// - the values of a tensor can be accessed via a multiindex which also has the type `[]usize`
        /// - the values of `dims` specify, which entries of the multiindex will be used.
        ///   (ex.: `&.{0, 1}` maps index-`0` and index-`1` onto the matrix, all other/further indices will be ingnored)
        /// - the order of `dims` specifies the memory layout.
        ///   (ex.: `&.{0, 1}` when scanning the allocated memory, index-`0` is scanned first before moving to the next value of index-`1`)
        /// - the first `sparsity` dimensions will be sparse.
        /// - only entries not equal to `zero` will take up memory in sparse dimensions.
        pub fn Sparse(comptime dims: []const usize, comptime sparsity: usize, comptime zero: Element) type {
            _ = Dims.from(dims);
            return sparse.Type(Element, Dims.from(dims[0..sparsity]), Dims.from(dims[sparsity..dims.len]), zero);
        }

        /// Returns a scalar result of the tensor operation:
        /// - `args` is an anonymous struct with the arguments of the operation.
        /// - `ew` is a function which is applied *elementwise*. It takes `args`, with each tensor swapped out for one of its elements,
        ///   as input and returns `Element` or `!Element`.
        /// - `red` is a function used to *reduce* the results of `ew` into a single scalar value. It takes two `Element`s as input and returns one `Element`.
        pub fn f(comptime red: anytype, comptime ew: anytype, args: anytype) Arg(Index, @TypeOf(args)).ErrorWrap(ew, Element) {
            const res = Dense(&.{}){
                .layout = .{
                    .size = undefined,
                    .incr = undefined,
                    .offset = 0,
                },
                .vals = MultiSlice(Element).stackInit(1),
            };
            const err = res.f(red, ew, args);
            assert(@TypeOf(if (@TypeOf(err) != void) try err else err) == void);
            return res.at(&.{});
        }
    };
}

test {
    _ = dense;
    _ = sparse;
    _ = simd;
}

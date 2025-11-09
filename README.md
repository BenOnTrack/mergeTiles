Merge multiple uncompressed MVT (v2.1) tiles into one:
 * - Group layers by name; for each name, union features from all sources.
 * - Build a unified keys[]/values[] dictionary; remap feature tags to new indices.
 * - If layer extents differ, decode geometry, rescale to target extent, and re-encode.
 * - Return a single Tile protobuf as Uint8Array (uncompressed).
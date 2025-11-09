/**
 * Merge multiple uncompressed MVT (v2.1) tiles into one:
 * - Group layers by name; for each name, union features from all sources.
 * - Build a unified keys[]/values[] dictionary; remap feature tags to new indices.
 * - If layer extents differ, decode geometry, rescale to target extent, and re-encode.
 * - Return a single Tile protobuf as Uint8Array (uncompressed).
 */
export async function mergeTiles(parts: Uint8Array[]): Promise<Uint8Array> {
  // Parse each tile into lightweight layer representations
  const tiles = parts.map(parseTileSafe);

  // Group layers with the same name across tiles (respect source order: bottom -> top)
  const layersByName = new Map<string, ParsedLayer[]>();
  parts.forEach((t, tileIdx) => {
    for (const lyr of t.layers) {
      const arr = layersByName.get(lyr.name) ?? [];
      arr.push({ ...lyr, _tileIndex: tileIdx });
      layersByName.set(lyr.name, arr);
    }
  });

  // Build merged layers (one per name)
  const mergedLayerBuffers: Uint8Array[] = [];
  for (const [name, layerGroup] of layersByName.entries()) {
    // Deterministic order: by tile index (bottom->top), preserve original feature order within each layer
    layerGroup.sort((a, b) => a._tileIndex - b._tileIndex);

    // Choose target extent (prefer the max; preserves precision when mixing sources)
    const targetExtent = Math.max(...layerGroup.map(l => l.extent ?? 4096));

    // Build unified dictionaries
    const keyIndex = new Map<string, number>();
    const keysOut: string[] = [];

    type ValKey = { t: string; v: string }; // canonical type+value string key
    const valIndex = new Map<string, number>();
    const valuesOut: ValueUnion[] = [];

    const addKey = (k: string) => {
      if (keyIndex.has(k)) return keyIndex.get(k)!;
      const idx = keysOut.length;
      keysOut.push(k);
      keyIndex.set(k, idx);
      return idx;
    };

    const addVal = (vu: ValueUnion) => {
      const k = canonicalValueKey(vu);
      if (valIndex.has(k)) return valIndex.get(k)!;
      const idx = valuesOut.length;
      valuesOut.push(vu);
      valIndex.set(k, idx);
      return idx;
    };

    // Prepare output features (remapped to unified dict and target extent)
    const featuresOut: EncodedFeature[] = [];
    let layerVersion = 2; // v2.x; we’ll take max encountered
    for (const lyr of layerGroup) {
      if (lyr.version && lyr.version > layerVersion) layerVersion = lyr.version;

      // Pre-build per-layer key/value maps for fast remap
      const perKey = new Map<number, number>();   // srcKeyIdx -> dstKeyIdx
      const perVal = new Map<number, number>();   // srcValIdx -> dstValIdx
      lyr.keys.forEach((k, i) => perKey.set(i, addKey(k)));
      lyr.values.forEach((v, i) => perVal.set(i, addVal(v)));

      const needRescale = (lyr.extent ?? targetExtent) !== targetExtent;
      const scale = needRescale ? (targetExtent / (lyr.extent ?? targetExtent)) : 1;

      for (const f of lyr.features) {
        // Remap tags: pairs of (keyIdx, valIdx)
        const remappedTags: number[] = [];
        for (let i = 0; i < f.tags.length; i += 2) {
          const kIdx = f.tags[i];
          const vIdx = f.tags[i + 1];
          remappedTags.push(perKey.get(kIdx)!);
          remappedTags.push(perVal.get(vIdx)!);
        }

        // Geometry: decode → (optional rescale) → encode
        let geomOut: Uint8Array;
        if (!needRescale) {
          // passthrough
          geomOut = f.geometryBytes; // already packed uint32 varints
        } else {
          const decoded = decodeGeometry(f.geometryBytes);           // commands with absolute coords
          const rescaled = rescaleGeometry(decoded, scale);          // multiply + round
          geomOut = encodeGeometry(rescaled);                        // back to delta ZigZag packed varints
        }

        featuresOut.push({
          id: f.id,             // bigint or number (we keep what we parsed)
          type: f.type,         // 1=POINT,2=LINESTRING,3=POLYGON
          tags: remappedTags,   // packed uint32
          geometryBytes: geomOut,
        });
      }
    }

    // Encode merged Layer
    const layerBuf = encodeLayer({
      version: layerVersion,
      name,
      keys: keysOut,
      values: valuesOut,
      extent: targetExtent,
      features: featuresOut,
    });

    mergedLayerBuffers.push(layerBuf);
  }

  // Encode final Tile: repeated Layer (field #3, wire type 2)
  const TAG_TILE_LAYER = 0x1a;
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (const layer of mergedLayerBuffers) {
    const lenVar = encodeVarint32(layer.length);
    const out = new Uint8Array(1 + lenVar.length + layer.length);
    out[0] = TAG_TILE_LAYER;
    out.set(lenVar, 1);
    out.set(layer, 1 + lenVar.length);
    chunks.push(out);
    total += out.length;
  }
  const mergedTile = new Uint8Array(total);
  {
    let o = 0;
    for (const c of chunks) {
      mergedTile.set(c, o);
      o += c.length;
    }
  }
  return mergedTile;
}

/* ------------------------------- Protobuf I/O ------------------------------- */

type ParsedTile = { layers: ParsedLayer[] };

type ParsedLayer = {
  version: number;
  name: string;
  keys: string[];
  values: ValueUnion[];
  extent: number;
  features: ParsedFeature[];
  _tileIndex?: number; // not encoded; used for sort
};

type ParsedFeature = {
  id?: bigint | number;
  tags: number[];             // packed uint32
  type: number;               // enum
  geometryBytes: Uint8Array;  // packed uint32
};

type ValueUnion =
  | { kind: 'string'; value: string }
  | { kind: 'float'; value: number }
  | { kind: 'double'; value: number }
  | { kind: 'int'; value: bigint }
  | { kind: 'uint'; value: bigint }
  | { kind: 'sint'; value: bigint }
  | { kind: 'bool'; value: boolean };

type EncodedFeature = ParsedFeature;

/** Safe tile parser (minimal fields needed for merge) */
function parseTileSafe(buf: Uint8Array): ParsedTile {
  const layers: ParsedLayer[] = [];
  let i = 0;
  while (i < buf.length) {
    const { value: tag, next: i1 } = decodeVarint32(buf, i);
    i = i1;
    const wt = tag & 0x7;
    const fn = tag >>> 3;
    if (wt === 2) {
      const { value: L, next: i2 } = decodeVarint32(buf, i);
      const end = i2 + L;
      if (fn === 3) {
        layers.push(parseLayer(buf.subarray(i2, end)));
      }
      i = end;
    } else {
      // skip varint / 32 / 64 as needed
      if (wt === 0) { const r = decodeVarint64(buf, i); i = r.next; }
      else if (wt === 5) { i += 4; }
      else if (wt === 1) { i += 8; }
      else break;
    }
  }
  return { layers };
}

function parseLayer(layer: Uint8Array): ParsedLayer {
  let i = 0;
  let version = 2;
  let name = '';
  const keys: string[] = [];
  const values: ValueUnion[] = [];
  const features: ParsedFeature[] = [];
  let extent = 4096;

  while (i < layer.length) {
    const { value: tag, next: i1 } = decodeVarint32(layer, i);
    i = i1;
    const wt = tag & 0x7;
    const fn = tag >>> 3;

    if (wt === 2) {
      const { value: L, next: i2 } = decodeVarint32(layer, i);
      const end = i2 + L;

      if (fn === 1) { // name
        name = utf8Decode(layer.subarray(i2, end));
      } else if (fn === 2) { // features (repeated)
        features.push(parseFeature(layer.subarray(i2, end)));
      } else if (fn === 3) { // keys (repeated string)
        keys.push(utf8Decode(layer.subarray(i2, end)));
      } else if (fn === 4) { // values (repeated Value msg)
        values.push(parseValue(layer.subarray(i2, end)));
      } else {
        // unknown length-delimited; skip
      }
      i = end;
    } else if (wt === 0) {
      // varint fields: version(15), extent(5)
      const r = decodeVarint64(layer, i);
      if (fn === 15) { version = Number(r.value); }
      else if (fn === 5) { extent = Number(r.value); }
      i = r.next;
    } else if (wt === 5) {
      i += 4;
    } else if (wt === 1) {
      i += 8;
    } else {
      break;
    }
  }

  return { version, name, keys, values, extent, features };
}

function parseFeature(fb: Uint8Array): ParsedFeature {
  let i = 0;
  let id: bigint | number | undefined;
  let type = 0;
  let tags: number[] = [];
  let geometryBytes = new Uint8Array(0);

  while (i < fb.length) {
    const { value: tag, next: i1 } = decodeVarint32(fb, i);
    i = i1;
    const wt = tag & 0x7;
    const fn = tag >>> 3;

    if (wt === 2) {
      const { value: L, next: i2 } = decodeVarint32(fb, i);
      const end = i2 + L;
      if (fn === 2) { // tags (packed uint32)
        tags = decodePackedUint32(fb.subarray(i2, end));
      } else if (fn === 4) { // geometry (packed uint32)
        geometryBytes = fb.subarray(i2, end);
      }
      i = end;
    } else if (wt === 0) {
      const r = decodeVarint64(fb, i);
      if (fn === 1) { id = r.value; }
      else if (fn === 3) { type = Number(r.value); }
      i = r.next;
    } else if (wt === 5) {
      i += 4;
    } else if (wt === 1) {
      i += 8;
    } else {
      break;
    }
  }

  return { id, type, tags, geometryBytes };
}

function parseValue(vb: Uint8Array): ValueUnion {
  let i = 0;
  while (i < vb.length) {
    const { value: tag, next: i1 } = decodeVarint32(vb, i);
    i = i1;
    const wt = tag & 0x7;
    const fn = tag >>> 3;

    if (fn === 1 && wt === 2) { // string_value
      const { value: L, next: i2 } = decodeVarint32(vb, i);
      const s = utf8Decode(vb.subarray(i2, i2 + L));
      return { kind: 'string', value: s };
    }
    if (fn === 2 && wt === 5) { // float (32-bit)
      const v = new DataView(vb.buffer, vb.byteOffset + i, 4).getFloat32(0, true);
      i += 4; return { kind: 'float', value: v };
    }
    if (fn === 3 && wt === 1) { // double (64-bit)
      const v = new DataView(vb.buffer, vb.byteOffset + i, 8).getFloat64(0, true);
      i += 8; return { kind: 'double', value: v };
    }
    if (fn === 4 && wt === 0) { const r = decodeVarint64(vb, i); i = r.next; return { kind: 'int', value: r.value }; }
    if (fn === 5 && wt === 0) { const r = decodeVarint64(vb, i); i = r.next; return { kind: 'uint', value: r.value }; }
    if (fn === 6 && wt === 0) { const r = decodeVarint64(vb, i); i = r.next; return { kind: 'sint', value: zigzagDecode64(r.value) }; }
    if (fn === 7 && wt === 0) { const r = decodeVarint64(vb, i); i = r.next; return { kind: 'bool', value: r.value !== 0n }; }

    // skip unknown
    if (wt === 2) { const { value: L, next } = decodeVarint32(vb, i); i = next + L; }
    else if (wt === 0) { const r = decodeVarint64(vb, i); i = r.next; }
    else if (wt === 5) { i += 4; }
    else if (wt === 1) { i += 8; }
    else break;
  }
  // Fallback (shouldn’t happen if spec is followed)
  return { kind: 'string', value: '' };
}

/* ----------------------------- Geometry helpers ---------------------------- */

type Command = { cmd: 'M' | 'L' | 'Z'; pts?: Array<[number, number]> };

/** Decode packed uint32 geometry into absolute coordinates per command. */
function decodeGeometry(geom: Uint8Array): Command[] {
  const cmds: Command[] = [];
  let i = 0;
  let x = 0, y = 0;

  while (i < geom.length) {
    const { value: cmdCount, next: i1 } = decodeVarint32(geom, i);
    i = i1;
    const cmd = cmdCount & 0x7;
    const count = cmdCount >>> 3;

    if (cmd === 1) { // MoveTo
      const pts: Array<[number, number]> = [];
      for (let k = 0; k < count; k++) {
        const dxVar = decodeVarint32(geom, i); i = dxVar.next;
        const dyVar = decodeVarint32(geom, i); i = dyVar.next;
        x += zigzagDecode32(dxVar.value);
        y += zigzagDecode32(dyVar.value);
        pts.push([x, y]);
      }
      cmds.push({ cmd: 'M', pts });
    } else if (cmd === 2) { // LineTo
      const pts: Array<[number, number]> = [];
      for (let k = 0; k < count; k++) {
        const dxVar = decodeVarint32(geom, i); i = dxVar.next;
        const dyVar = decodeVarint32(geom, i); i = dyVar.next;
        x += zigzagDecode32(dxVar.value);
        y += zigzagDecode32(dyVar.value);
        pts.push([x, y]);
      }
      cmds.push({ cmd: 'L', pts });
    } else if (cmd === 7) { // ClosePath
      cmds.push({ cmd: 'Z' });
      // No parameters
    } else {
      // Unknown command id per spec: bail safely
      break;
    }
  }

  return cmds;
}

/** Rescale absolute coordinates by factor; round to nearest int. */
function rescaleGeometry(commands: Command[], scale: number): Command[] {
  if (scale === 1) return commands;
  return commands.map(c => {
    if (!c.pts) return c;
    const pts = c.pts.map(([px, py]) => [Math.round(px * scale), Math.round(py * scale)] as [number, number]);
    return { cmd: c.cmd, pts };
  });
}

/** Encode commands with absolute coords back into packed uint32 delta ZigZag stream. */
function encodeGeometry(commands: Command[]): Uint8Array {
  const out: number[] = [];
  let x = 0, y = 0;

  const emitCmd = (id: number, count: number) => out.push((count << 3) | (id & 0x7));
  const emitParam = (v: number) => out.push(...encodeVarint32(zigzagEncode32(v)));

  for (const c of commands) {
    if (c.cmd === 'M' && c.pts && c.pts.length) {
      emitCmd(1, c.pts.length);
      for (const [ax, ay] of c.pts) {
        emitParam(ax - x);
        emitParam(ay - y);
        x = ax; y = ay;
      }
    } else if (c.cmd === 'L' && c.pts && c.pts.length) {
      emitCmd(2, c.pts.length);
      for (const [ax, ay] of c.pts) {
        emitParam(ax - x);
        emitParam(ay - y);
        x = ax; y = ay;
      }
    } else if (c.cmd === 'Z') {
      emitCmd(7, 1);
    }
  }

  return new Uint8Array(out);
}

/* ------------------------------ Encoders (PB) ------------------------------- */

function encodeLayer(l: {
  version: number;
  name: string;
  keys: string[];
  values: ValueUnion[];
  extent: number;
  features: EncodedFeature[];
}): Uint8Array {
  const chunks: Uint8Array[] = [];
  let total = 0;

  // required uint32 version = 15 [ default = 1 ];
  chunks.push(encodeFieldVarint(15, l.version)); total += chunks[chunks.length - 1].length;

  // required string name = 1;
  chunks.push(encodeFieldString(1, l.name)); total += chunks[chunks.length - 1].length;

  // repeated Feature features = 2;
  for (const f of l.features) {
    const fbuf = encodeFeature(f);
    chunks.push(encodeFieldBytes(2, fbuf));
    total += chunks[chunks.length - 1].length;
  }

  // repeated string keys = 3;
  for (const k of l.keys) {
    const kbuf = utf8Encode(k);
    chunks.push(encodeFieldBytes(3, kbuf));
    total += chunks[chunks.length - 1].length;
  }

  // repeated Value values = 4;
  for (const v of l.values) {
    const vbuf = encodeValue(v);
    chunks.push(encodeFieldBytes(4, vbuf));
    total += chunks[chunks.length - 1].length;
  }

  // optional uint32 extent = 5; (required by spec, but wire is optional)
  chunks.push(encodeFieldVarint(5, l.extent)); total += chunks[chunks.length - 1].length;

  const out = new Uint8Array(total);
  { let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } }
  return out;
}

function encodeFeature(f: EncodedFeature): Uint8Array {
  const chunks: Uint8Array[] = [];
  let total = 0;

  // optional uint64 id = 1
  if (typeof f.id !== 'undefined') {
    chunks.push(encodeFieldVarint64(1, typeof f.id === 'bigint' ? f.id : BigInt(f.id)));
    total += chunks[chunks.length - 1].length;
  }

  // repeated uint32 tags = 2 [packed = true]
  const tagsPacked = new Uint8Array(encodePackedUint32(f.tags));
  chunks.push(encodeFieldBytes(2, tagsPacked)); total += chunks[chunks.length - 1].length;

  // optional GeomType type = 3
  chunks.push(encodeFieldVarint(3, f.type)); total += chunks[chunks.length - 1].length;

  // repeated uint32 geometry = 4 [packed = true]
  chunks.push(encodeFieldBytes(4, f.geometryBytes)); total += chunks[chunks.length - 1].length;

  const out = new Uint8Array(total);
  { let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } }
  return out;
}

function encodeValue(v: ValueUnion): Uint8Array {
  // oneof string/float/double/int/uint/sint/bool
  if (v.kind === 'string') return encodeFieldBytes(1, utf8Encode(v.value));
  if (v.kind === 'float') {
    const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v.value, true);
    return encodeFieldFixed32(2, b);
  }
  if (v.kind === 'double') {
    const b = new Uint8Array(8); new DataView(b.buffer).setFloat64(0, v.value, true);
    return encodeFieldFixed64(3, b);
  }
  if (v.kind === 'int')   return encodeFieldVarint64(4, v.value);
  if (v.kind === 'uint')  return encodeFieldVarint64(5, v.value);
  if (v.kind === 'sint')  return encodeFieldVarint64(6, zigzagEncode64(v.value));
  if (v.kind === 'bool')  return encodeFieldVarint(7, v.value ? 1 : 0);
  // default
  return encodeFieldBytes(1, utf8Encode(''));
}

/* ------------------------------ PB primitives ------------------------------- */

function encodeFieldVarint(fieldNum: number, value32: number): Uint8Array {
  const tag = (fieldNum << 3) | 0; // wire=0
  const v = encodeVarint32(value32 >>> 0);
  const out = new Uint8Array(1 + v.length);
  out[0] = tag; out.set(v, 1);
  return out;
}
function encodeFieldVarint64(fieldNum: number, value64: bigint): Uint8Array {
  const tag = (fieldNum << 3) | 0;
  const v = encodeVarint64(value64);
  const out = new Uint8Array(1 + v.length);
  out[0] = tag; out.set(v, 1);
  return out;
}
function encodeFieldFixed32(fieldNum: number, bytes4: Uint8Array): Uint8Array {
  const tag = (fieldNum << 3) | 5;
  const out = new Uint8Array(1 + 4);
  out[0] = tag; out.set(bytes4, 1);
  return out;
}
function encodeFieldFixed64(fieldNum: number, bytes8: Uint8Array): Uint8Array {
  const tag = (fieldNum << 3) | 1;
  const out = new Uint8Array(1 + 8);
  out[0] = tag; out.set(bytes8, 1);
  return out;
}
function encodeFieldBytes(fieldNum: number, bytes: Uint8Array): Uint8Array {
  const tag = (fieldNum << 3) | 2;
  const len = encodeVarint32(bytes.length);
  const out = new Uint8Array(1 + len.length + bytes.length);
  out[0] = tag; out.set(len, 1); out.set(bytes, 1 + len.length);
  return out;
}

function decodeVarint32(buf: Uint8Array, o: number): { value: number; next: number } {
  let x = 0, s = 0, i = o;
  for (; i < buf.length; i++) {
    const b = buf[i];
    if (b < 0x80) { x |= b << s; return { value: x >>> 0, next: i + 1 }; }
    x |= (b & 0x7f) << s; s += 7;
  }
  return { value: x >>> 0, next: i };
}
function encodeVarint32(v: number): Uint8Array {
  const out: number[] = [];
  let x = v >>> 0;
  while (x >= 0x80) { out.push((x & 0x7f) | 0x80); x >>>= 7; }
  out.push(x);
  return new Uint8Array(out);
}

function decodeVarint64(buf: Uint8Array, o: number): { value: bigint; next: number } {
  let x = 0n, s = 0n, i = o;
  for (; i < buf.length; i++) {
    const b = BigInt(buf[i]);
    if (b < 0x80n) { x |= (b << s); return { value: x, next: i + 1 }; }
    x |= ((b & 0x7fn) << s); s += 7n;
  }
  return { value: x, next: i };
}
function encodeVarint64(v: bigint): Uint8Array {
  const out: number[] = [];
  let x = v;
  while (x >= 0x80n) { out.push(Number((x & 0x7fn) | 0x80n)); x >>= 7n; }
  out.push(Number(x));
  return new Uint8Array(out);
}

function zigzagEncode32(n: number): number { return (n << 1) ^ (n >> 31); }
function zigzagDecode32(n: number): number { return (n >>> 1) ^ -(n & 1); }

function zigzagEncode64(n: bigint): bigint { return (n << 1n) ^ (n >> 63n); }
function zigzagDecode64(n: bigint): bigint { return (n >> 1n) ^ (-(n & 1n)); }

function utf8Decode(u8: Uint8Array): string { return new TextDecoder('utf-8').decode(u8); }
function utf8Encode(s: string): Uint8Array { return new TextEncoder().encode(s); }

function decodePackedUint32(buf: Uint8Array): number[] {
  const out: number[] = [];
  let i = 0;
  while (i < buf.length) {
    const r = decodeVarint32(buf, i);
    out.push(r.value);
    i = r.next;
  }
  return out;
}
function encodePackedUint32(arr: number[]): Uint8Array {
  const out: number[] = [];
  for (const v of arr) out.push(...encodeVarint32(v >>> 0));
  return new Uint8Array(out);
}

/* -------------------------- Value canonicalization ------------------------- */

function canonicalValueKey(v: ValueUnion): string {
  switch (v.kind) {
    case 'string': return `s:${v.value}`;
    case 'float':  return `f:${v.value}`;
    case 'double': return `d:${v.value}`;
    case 'int':    return `i:${v.value.toString()}`;
    case 'uint':   return `u:${v.value.toString()}`;
    case 'sint':   return `n:${v.value.toString()}`;
    case 'bool':   return `b:${v.value ? 1 : 0}`;
  }
}

function encodeFieldString(fieldNum: number, s: string): Uint8Array {
  return encodeFieldBytes(fieldNum, utf8Encode(s));
}
import React, { useEffect, useMemo, useRef, useState } from "react";

const API_URL = "http://localhost:8000/extract"; // change if needed

const SPEC_OPTIONS = [1,2,3,4,5,6,7,8,9,10,11,12];

const RULE_SPEC = {
  1:  { shape: "circle", color: "red",    use: "stroke" },
  2:  { shape: "circle", color: "purple", use: "stroke" },
  3:  { shape: "circle", color: "green",  use: "stroke" },
  4:  { shape: "circle", color: "orange", use: "stroke" },
  5:  { shape: "circle", color: "black",  use: "stroke" },

  6:  { shape: "rect",   color: "red",    use: "stroke" },
  7:  { shape: "rect",   color: "black",  use: "stroke" },
  8:  { shape: "rect",   color: "green",  use: "stroke" },

  9:  { shape: "rect",   color: "yellow", use: "fill" },

  10: { shape: "poly4",  color: "green",  use: "stroke" },
  11: { shape: "poly5",  color: "green",  use: "stroke" },
  12: { shape: "poly6",  color: "green",  use: "stroke" },
};

function specLabel(id) {
  const s = RULE_SPEC[id];
  return s ? `${id} — ${s.shape} / ${s.color} / ${s.use}` : String(id);
}

function toSpecsCsv(specsSet) {
  return Array.from(specsSet).sort((a,b)=>a-b).join(",");
}

export default function App() {
  const [file, setFile] = useState(null);
  const [imgUrl, setImgUrl] = useState("");
  const [imgNatural, setImgNatural] = useState({ w: 0, h: 0 });
  const [specs, setSpecs] = useState(() => new Set([6]));
  const [loading, setLoading] = useState(false);

  const [resp, setResp] = useState(null);
  const tags = useMemo(() => {
    // supports either {result:{extracted_tml_tags:...}} or {extracted_tml_tags:...}
    if (!resp) return [];
    return resp?.result?.extracted_tml_tags ?? resp?.extracted_tml_tags ?? [];
  }, [resp]);

  const [hoveredIdx, setHoveredIdx] = useState(-1);

  const imgRef = useRef(null);

  // create object URL for preview
  useEffect(() => {
    if (!file) {
      setImgUrl("");
      setResp(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setImgUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function toggleSpec(s) {
    setSpecs(prev => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  async function runExtract() {
    if (!file) return;

    setLoading(true);
    setResp(null);
    setHoveredIdx(-1);

    try {
      const fd = new FormData();
      fd.append("file", file);

      const qs = new URLSearchParams();
      qs.set("specs_allowed", toSpecsCsv(specs));

      const r = await fetch(`${API_URL}?${qs.toString()}`, {
        method: "POST",
        body: fd,
      });

      if (!r.ok) {
        const text = await r.text();
        throw new Error(`HTTP ${r.status}: ${text}`);
      }

      const data = await r.json();
      setResp(data);
    } catch (e) {
      setResp({ error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  function onImgLoad() {
    const img = imgRef.current;
    if (!img) return;
    setImgNatural({ w: img.naturalWidth, h: img.naturalHeight });
  }

  // map bbox from image coordinate system to displayed size (responsive)
  function bboxToViewBox(bboxOrig, displayW, displayH) {
    // bbox_orig: [x, y, w, h] in ORIGINAL image pixels
    // We render SVG with viewBox in original pixel coords, then scale automatically.
    // So we can just return original coords; scaling is handled by SVG viewBox.
    const [x, y, w, h] = bboxOrig;
    return { x, y, w, h };
  }

  const canRenderOverlay = imgNatural.w > 0 && imgNatural.h > 0 && imgUrl;

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial", padding: 16, maxWidth: 1200, margin: "0 auto" }}>
      <h2 style={{ marginTop: 0 }}>TML Extract Demo</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
        {/* Controls */}
        <div style={{ width: 400, border: "1px solid #ddd", borderRadius: 12, padding: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Input</div>

          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />

          <div style={{ marginTop: 12, fontWeight: 600 }}>Specs</div>
          <div style={{
            marginTop: 8,
            display: "grid",
            gridTemplateColumns: "repeat(6, minmax(0, 1fr))",
            gap: "6px 10px"
          }}>
            {SPEC_OPTIONS.map((s) => {
              const info = RULE_SPEC[s]; // {shape,color,use} or undefined
              return (
                <label
                  key={s}
                  style={{
                    cursor: "pointer",
                    userSelect: "none",
                    display: "flex",
                    gap: 8,
                    alignItems: "flex-start",
                  }}
                  title={info ? `${info.shape}, ${info.color}, ${info.use}` : ""}
                >
                  <input
                    type="checkbox"
                    checked={specs.has(s)}
                    onChange={() => toggleSpec(s)}
                    style={{ marginTop: 3 }}
                  />
                  <div style={{ lineHeight: 1.15 }}>
                    <div style={{ fontWeight: 600 }}>{s}</div>
                    {info && (
                      <div style={{ fontSize: 12, color: "#666" }}>
                        {info.shape}<br/>
                        {info.color}<br/>
                        {info.use}
                      </div>
                    )}
                  </div>
                </label>
              );
            })}
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button
              onClick={() => setSpecs(new Set(SPEC_OPTIONS))}
              style={btnStyle}
              type="button"
            >
              All
            </button>
            <button
              onClick={() => setSpecs(new Set())}
              style={btnStyle}
              type="button"
            >
              None
            </button>
          </div>

          <button
            onClick={runExtract}
            disabled={!file || loading}
            style={{ ...btnStyle, width: "100%", marginTop: 12 }}
            type="button"
          >
            {loading ? "Running..." : "Run /extract"}
          </button>

          <div style={{ marginTop: 10, fontSize: 12, color: "#666" }}>
            specs_allowed = <code>{toSpecsCsv(specs) || "(empty)"}</code>
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Response JSON</div>
            <pre style={{
              margin: 0,
              maxHeight: 260,
              overflow: "auto",
              background: "#fafafa",
              border: "1px solid #eee",
              borderRadius: 10,
              padding: 10,
              fontSize: 12
            }}>
              {resp ? JSON.stringify(resp, null, 2) : "—"}
            </pre>
          </div>
        </div>

        {/* Viewer */}
        <div style={{ flex: 1, minWidth: 380, border: "1px solid #ddd", borderRadius: 12, padding: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Visualization</div>

          {!imgUrl && <div style={{ color: "#666" }}>Upload an image to start.</div>}

          {imgUrl && (
            <div style={{ position: "relative", display: "inline-block", borderRadius: 12, overflow: "hidden", border: "1px solid #eee" }}>
              {/* base image */}
              <img
                ref={imgRef}
                src={imgUrl}
                alt="upload"
                onLoad={onImgLoad}
                style={{ display: "block", maxWidth: "100%", height: "auto" }}
              />

              {/* overlay (SVG) */}
              {canRenderOverlay && (
                <svg
                  viewBox={`0 0 ${imgNatural.w} ${imgNatural.h}`}
                  style={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    width: "100%",
                    height: "100%",
                    pointerEvents: "auto",
                  }}
                >
                  {tags.map((t, idx) => {
                    const bbox = t.bbox_orig;
                    if (!bbox || bbox.length !== 4) return null;
                    const { x, y, w, h } = bboxToViewBox(bbox, imgNatural.w, imgNatural.h);

                    const tagId = String(t.tag_id ?? "");
                    const isHover = idx === hoveredIdx;

                    // base opacity for all; hover increases
                    const fillOpacity = isHover ? 0.18 : 0.06;
                    const strokeOpacity = isHover ? 0.95 : 0.55;

                    return (
                      <g
                        key={idx}
                        onMouseEnter={() => setHoveredIdx(idx)}
                        onMouseLeave={() => setHoveredIdx(-1)}
                        style={{ cursor: "pointer" }}
                      >
                        <rect
                          x={x}
                          y={y}
                          width={w}
                          height={h}
                          fill={`rgba(0, 255, 0, ${fillOpacity})`}
                          stroke={`rgba(0, 255, 0, ${strokeOpacity})`}
                          strokeWidth={isHover ? 3 : 2}
                        />
                        {/* label background */}
                        <rect
                          x={x}
                          y={Math.max(0, y - 22)}
                          width={Math.max(24, tagId.length * 10 + 12)}
                          height={22}
                          rx={6}
                          fill={`rgba(0,0,0, ${isHover ? 0.65 : 0.45})`}
                        />
                        <text
                          x={x + 8}
                          y={Math.max(16, y - 7)}
                          fill={`rgba(255,255,255, ${isHover ? 1.0 : 0.9})`}
                          fontSize={14}
                          fontFamily="system-ui, -apple-system, Segoe UI, Roboto, Arial"
                          style={{ userSelect: "none" }}
                        >
                          {tagId}
                        </text>

                        {/* center dot */}
                        {Number.isFinite(t.center_x) && Number.isFinite(t.center_y) && (
                          <circle
                            cx={t.center_x}
                            cy={t.center_y}
                            r={isHover ? 4 : 3}
                            fill={`rgba(255, 205, 0, ${isHover ? 0.95 : 0.7})`}
                          />
                        )}
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          )}

          {tags?.length > 0 && (
            <div style={{ marginTop: 10, fontSize: 12, color: "#666" }}>
              {tags.length} boxes (hover to highlight)
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const btnStyle = {
  padding: "10px 12px",
  borderRadius: 10,
  border: "1px solid #ccc",
  background: "#fafafa",
  cursor: "pointer",
};

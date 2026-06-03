#!/usr/bin/env python3
"""
Find the coordinates of a 0.5-mile radius circle whose center maximizes the
intersecting area of MIXED USE, URBAN VILLAGE, and MULTIFAMILY RESIDENTIAL
land-use polygons.

Algorithm: Differential Evolution (global optimizer) followed by L-BFGS-B
polishing, both operating on projected UTM 10N coordinates for accurate
distance and area measurements.

Dependencies:
    pip install shapely pyproj scipy numpy folium
"""

import json
import time
import webbrowser
from pathlib import Path

import numpy as np
from shapely import contains_xy as _shapely_contains_xy
import folium
from folium import Element
from shapely.geometry import shape, Point
from shapely.ops import transform
from shapely import STRtree, make_valid
import pyproj
from scipy.optimize import differential_evolution

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEOJSON_PATH = "example_issaquah.geojson"
RADIUS_MILES = 0.5
RADIUS_METERS = RADIUS_MILES * 1609.344

TARGET_LANDUSES = {"MIXED USE", "URBAN VILLAGE", "MULTIFAMILY RESIDENTIAL"}

# Set to True to restrict the optimizer to the LineString corridor (1-D search
# along the arc-length of the line).  False runs an unconstrained 2-D search
# over the full dataset bounding box regardless of whether LineStrings exist.
USE_PATH_CONSTRAINT = True

# If True, the objective function weights area elements by distance to center of radius
BIAS_TOWARDS_CENTER = False  

# Max distance (m) from candidate center to a LineString to satisfy path_constraint.
# A point almost never lies *exactly* on a 1-D line in floating-point space, so
# we use a small snap buffer instead
PATH_SNAP_METERS = 25

# Raster cell size (m) for gradient-weighted area integration.
# Smaller cells are more accurate but slower. 20m gives <1% error for typical polygon sizes.
GRADIENT_CELL_SIZE = 20.0

# UTM Zone 10N is the correct projected CRS for the Issaquah/Bellevue area
SRC_CRS = "EPSG:4326"   # WGS84 geographic (lon/lat)
DST_CRS = "EPSG:32610"  # UTM 10N (meters)

# Colors per land-use type (fill, border)
LANDUSE_STYLE = {
    "MIXED USE":              {"color": "#E65C00", "fillColor": "#FF6B35"},
    "URBAN VILLAGE":          {"color": "#5A1A7A", "fillColor": "#9B4DCA"},
    "MULTIFAMILY RESIDENTIAL": {"color": "#0D5FA6", "fillColor": "#2196F3"},
}


# ---------------------------------------------------------------------------
# Gradient-weighted area helper
# ---------------------------------------------------------------------------

def _gradient_weighted_area(clipped_poly, center_pt):
    """
    Numerically integrate max(0, RADIUS_METERS - r) dA over clipped_poly,
    where r is the distance from center_pt to each area element.

    Rasterizes the polygon bounding box at GRADIENT_CELL_SIZE resolution and
    uses shapely's vectorized contains_xy to classify interior cells in one
    call, avoiding any Python-level point-in-polygon loop.
    """
    minx, miny, maxx, maxy = clipped_poly.bounds
    cx, cy = center_pt.x, center_pt.y

    xs = np.arange(minx + GRADIENT_CELL_SIZE / 2, maxx, GRADIENT_CELL_SIZE)
    ys = np.arange(miny + GRADIENT_CELL_SIZE / 2, maxy, GRADIENT_CELL_SIZE)
    if xs.size == 0 or ys.size == 0:
        return 0.0

    px, py = np.meshgrid(xs, ys)
    px, py = px.ravel(), py.ravel()

    mask = _shapely_contains_xy(clipped_poly, px, py)
    if not mask.any():
        return 0.0

    dists = np.hypot(px[mask] - cx, py[mask] - cy)
    return float(np.maximum(0.0, RADIUS_METERS - dists).sum()) * GRADIENT_CELL_SIZE ** 2


# ---------------------------------------------------------------------------
# Load and project geometries
# ---------------------------------------------------------------------------

def load_target_polygons(path: str):
    """
    Load GeoJSON, filter by target land use.
    Returns:
        proj             – pyproj transformer callable (lon/lat → easting/northing)
        target_utm       – list of Shapely geometries in UTM 10N (polygons only)
        target_features  – list of raw GeoJSON feature dicts (WGS84, for rendering)
        linestring_utm   – list of projected LineString geometries for path constraint
    """
    print(f"Loading {path} ...")
    with open(path) as f:
        data = json.load(f)

    total = len(data["features"])

    # Separate polygon-type target features from LineString features
    target_features = [
        feat for feat in data["features"]
        if feat["geometry"]["type"] not in ("LineString", "MultiLineString")
        and feat["properties"]["Landuse"] in TARGET_LANDUSES
    ]
    linestring_features = [
        feat for feat in data["features"]
        if feat["geometry"]["type"] in ("LineString", "MultiLineString")
    ]
    print(f"  Total features   : {total}")
    print(f"  Target features  : {len(target_features)}  "
          f"({', '.join(sorted(TARGET_LANDUSES))})")
    print(f"  Path LineStrings : {len(linestring_features)}")

    # always_xy=True keeps (lon, lat) / (easting, northing) ordering throughout
    transformer = pyproj.Transformer.from_crs(SRC_CRS, DST_CRS, always_xy=True)
    proj = transformer.transform

    # make_valid repairs self-intersections and degenerate rings that cause
    # TopologyException during intersection ops
    target_utm = [
        make_valid(transform(proj, shape(feat["geometry"])))
        for feat in target_features
    ]
    linestring_utm = [
        transform(proj, shape(feat["geometry"]))
        for feat in linestring_features
    ]
    return proj, target_utm, target_features, linestring_utm


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def make_objective(proj, target_utm, linestring_utm):
    """Return a callable that computes -eligible_area(lon, lat)."""
    tree = STRtree(target_utm)
    # Separate STRtree for fast path-constraint lookups
    path_tree = STRtree(linestring_utm) if linestring_utm else None

    def eligible_area(lon: float, lat: float,
                      bias_towards_center: bool = False,
                      path_constraint: bool = False) -> float:
        """Total area (m^2) of target polygons intersecting a 0.5-mi circle."""
        cx, cy = proj(lon, lat)          # UTM coordinates of candidate center
        center_pt = Point(cx, cy)        # UTM Point — same CRS as all geometries

        # Hard path constraint: center must be within PATH_SNAP_METERS of a
        # LineString.  A floating-point point almost never lies *exactly* on a
        # 1-D line, so we test with a small buffer rather than a bare intersects.
        if path_constraint and path_tree is not None:
            snap_buf = center_pt.buffer(PATH_SNAP_METERS)
            on_path = len(path_tree.query(snap_buf, predicate="intersects")) > 0
            if not on_path:
                return 0.0

        circle = center_pt.buffer(RADIUS_METERS, quad_segs=64)

        # STRtree.query with predicate skips the two-step bounding-box filter
        try:
            candidate_idx = tree.query(circle, predicate="intersects")
        except TypeError:
            # Shapely <2.0 fallback: query returns bbox candidates; filter manually
            candidate_idx = [
                i for i in tree.query(circle)
                if target_utm[i].intersects(circle)
            ]

        if not bias_towards_center:
            return sum(target_utm[i].intersection(circle).area for i in candidate_idx)
        else:
            # Integrate (RADIUS_METERS - r) dA over each clipped polygon, where
            # r is the per-point distance to center
            # continuous linear gradient such that land closer to the center carries more weight.
            return sum(
                _gradient_weighted_area(target_utm[i].intersection(circle), center_pt)
                for i in candidate_idx
            )

    def neg_eligible_area(coords):
        return -eligible_area(coords[0], coords[1])

    return eligible_area, neg_eligible_area


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(target_features, lon_opt, lat_opt, area_m2,
              output_path="station_optimization.html"):
    """
    Build an interactive Leaflet map showing:
      • Eligible land-use polygons, color-coded and layer-togglable by type
      • 0.5-mile search radius circle around the optimal center
      • Marker at the optimal center with a summary popup
      • Legend and layer control

    Saves the map to output_path and opens it in the default browser.
    """
    area_acres = area_m2 / 4_046.856_422_4
    area_mi2   = area_m2 / (1_609.344 ** 2)

    # Base map centered on the optimal point
    m = folium.Map(
        location=[lat_opt, lon_opt],
        zoom_start=14,
        tiles="CartoDB positron",
        attr="CartoDB",
    )

    # --- Eligible polygons, one FeatureGroup per land-use type ---------------
    groups = {}
    for lu in sorted(TARGET_LANDUSES):
        style = LANDUSE_STYLE[lu]
        fg = folium.FeatureGroup(name=lu.title(), show=True)
        groups[lu] = (fg, style)
        fg.add_to(m)

    for feat in target_features:
        lu = feat["properties"]["Landuse"]
        fg, style = groups[lu]
        folium.GeoJson(
            feat,
            style_function=lambda _, s=style: {
                "fillColor": s["fillColor"],
                "color":     s["color"],
                "weight":    1.2,
                "fillOpacity": 0.55,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["Landuse"],
                aliases=["Land Use:"],
                localize=True,
            ),
        ).add_to(fg)

    # --- Optimal circle -------------------------------------------------------
    circle_group = folium.FeatureGroup(name="0.5-mi Radius", show=True)
    folium.Circle(
        location=[lat_opt, lon_opt],
        radius=RADIUS_METERS,
        color="#CC0000",
        weight=2.5,
        fill=True,
        fill_color="#FF4444",
        fill_opacity=0.08,
        tooltip=f"0.5-mi radius ({RADIUS_METERS:.0f} m)",
    ).add_to(circle_group)
    circle_group.add_to(m)

    # --- Optimal center marker ------------------------------------------------
    marker_group = folium.FeatureGroup(name="Optimal Center", show=True)
    popup_html = (
        f"<div style='font-family:sans-serif;min-width:190px'>"
        f"<b style='font-size:14px'>Optimal Station Location</b><br><br>"
        f"<table style='border-spacing:4px 2px'>"
        f"<tr><td><b>Longitude</b></td><td>{lon_opt:.7f}</td></tr>"
        f"<tr><td><b>Latitude</b></td><td>{lat_opt:.7f}</td></tr>"
        f"<tr><td><b>Eligible area</b></td><td>{area_acres:,.1f} ac</td></tr>"
        f"<tr><td><b>&nbsp;</b></td><td>{area_mi2:.4f} sq mi</td></tr>"
        f"<tr><td><b>Coverage</b></td><td>"
        f"{area_mi2 / (np.pi * RADIUS_MILES**2) * 100:.1f}% of circle</td></tr>"
        f"</table></div>"
    )
    folium.Marker(
        location=[lat_opt, lon_opt],
        popup=folium.Popup(popup_html, max_width=260),
        tooltip="Optimal center — click for details",
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(marker_group)
    marker_group.add_to(m)

    # --- Layer control --------------------------------------------------------
    folium.LayerControl(collapsed=False).add_to(m)

    # --- Legend (fixed bottom-left) ------------------------------------------
    legend_items = ""
    for lu in sorted(TARGET_LANDUSES):
        fill = LANDUSE_STYLE[lu]["fillColor"]
        border = LANDUSE_STYLE[lu]["color"]
        legend_items += (
            f"<div style='display:flex;align-items:center;margin:3px 0'>"
            f"<span style='background:{fill};width:14px;height:14px;"
            f"display:inline-block;margin-right:8px;"
            f"border:1px solid {border};opacity:0.8'></span>"
            f"{lu.title()}</div>"
        )
    legend_html = f"""
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        border: 1px solid #bbb; font-family: sans-serif; font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
      <b style="font-size:14px">Land Use</b>
      <div style="margin-top:6px">{legend_items}</div>
      <div style="margin-top:6px;display:flex;align-items:center">
        <span style="background:#FF4444;width:14px;height:3px;display:inline-block;
              margin-right:8px;border-top:2px solid #CC0000"></span>
        0.5-mi Radius
      </div>
      <div style="margin-top:4px;display:flex;align-items:center">
        <span style="color:#CC0000;font-size:16px;margin-right:6px">&#9733;</span>
        Optimal Center
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    # --- Save and open --------------------------------------------------------
    out = Path(output_path).resolve()
    m.save(str(out))
    print(f"\nMap saved to: {out}")
    webbrowser.open(out.as_uri())
    return str(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    proj, target_utm, target_features, linestring_utm = load_target_polygons(GEOJSON_PATH)
    eligible_area, neg_eligible_area = make_objective(proj, target_utm, linestring_utm)

    # Inverse transformer: UTM 10N → WGS84, used to recover (lon, lat) from a
    # point interpolated along the LineString in projected space.
    inv_proj = pyproj.Transformer.from_crs(DST_CRS, SRC_CRS, always_xy=True).transform

    use_path_constraint = USE_PATH_CONSTRAINT and bool(linestring_utm)

    if use_path_constraint:
        # 1-D search: the optimizer controls arc-length fraction t ∈ [0, 1]
        # along the LineString.  Every candidate is guaranteed to lie on the
        # path, which avoids the infeasible-region problem that a hard 2-D
        # constraint causes when the LineString covers <<1% of the search box.
        line_utm = linestring_utm[0]

        def neg_area_on_path(params):
            t = float(params[0])
            pt = line_utm.interpolate(t * line_utm.length)
            lon, lat = inv_proj(pt.x, pt.y)
            return -eligible_area(lon, lat, path_constraint=False)

        opt_func   = neg_area_on_path
        opt_bounds = [(0.0, 1.0)]
        print(f"\nOptimizing along path LineString (1-D, t = arc-length fraction) ...")
        print(f"  LineString length : {line_utm.length:.1f} m")
    else:
        # Unconstrained 2-D search over the full dataset bounding box
        opt_func   = neg_eligible_area
        opt_bounds = [(-122.100470, -121.985731), (47.509154, 47.577366)]
        print(f"\nOptimizing with Differential Evolution (2-D, no path constraint) ...")
        print(f"  Search bounds : lon {opt_bounds[0]}")
        print(f"                  lat {opt_bounds[1]}")

    print(f"  Circle radius : {RADIUS_MILES} mi = {RADIUS_METERS:.2f} m")
    print()

    t0 = time.time()
    result = differential_evolution(
        opt_func,
        opt_bounds,
        seed=42,
        maxiter=1000,
        popsize=20,
        tol=1e-8,
        mutation=(0.5, 1.5),
        recombination=0.9,
        disp=True,
        polish=True,
    )
    elapsed = time.time() - t0

    # Recover (lon, lat) from the optimization result
    if use_path_constraint:
        t_opt = float(result.x[0])
        pt_opt = line_utm.interpolate(t_opt * line_utm.length)
        lon_opt, lat_opt = inv_proj(pt_opt.x, pt_opt.y)
    else:
        lon_opt, lat_opt = result.x

    # ---------------------------------------------------------------------------
    # Report results
    # ---------------------------------------------------------------------------
    # The optimizer may have used a weighted objective (bias_towards_center),
    # so recompute the plain area at the optimal point for human-readable output.
    area_m2    = eligible_area(lon_opt, lat_opt, bias_towards_center=False,
                               path_constraint=False)
    area_acres = area_m2 / 4_046.856_422_4
    area_mi2   = area_m2 / (1_609.344 ** 2)
    circle_area_mi2 = np.pi * RADIUS_MILES ** 2

    print()
    print("=" * 60)
    print("  OPTIMIZATION RESULT")
    print("=" * 60)
    print(f"  Status             : {result.message}")
    print(f"  Elapsed            : {elapsed:.1f} s  ({result.nfev} evaluations)")
    print()
    print(f"  Optimal center")
    print(f"    Longitude        : {lon_opt:.7f}")
    print(f"    Latitude         : {lat_opt:.7f}")
    print()
    print(f"  Eligible area within {RADIUS_MILES}-mi circle")
    print(f"    Square meters    : {area_m2:>14,.1f} sq m")
    print(f"    Acres            : {area_acres:>14,.2f} ac")
    print(f"    Square miles     : {area_mi2:>14.5f} sq mi")
    print()
    print(f"  Circle total area  : {circle_area_mi2:.5f} sq mi")
    print(f"  Coverage fraction  : {area_mi2 / circle_area_mi2 * 100:.1f}%")
    print("=" * 60)

    visualize(target_features, lon_opt, lat_opt, area_m2)

    return lon_opt, lat_opt, area_m2


if __name__ == "__main__":
    main()

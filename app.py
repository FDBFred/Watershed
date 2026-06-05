import streamlit as st
import gpxpy
import numpy as np
import rasterio
from rasterio.transform import rowcol
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import requests
import tempfile
import os
import math
import io
from numba import njit
from rdp import rdp

# --- UI Framework Configuration ---
st.set_page_config(page_title="Professional Viewshed Engine", layout="wide")
st.title("🗺️ Professional Trail Viewshed Engine")
st.markdown("High-performance terrain ray-caster providing standalone interactive 3D WebGL meshes and clean 2D spatial overlays.")

# --- High-Performance Core Engine (Numba JIT Machine Compiled) ---

@njit(fastmath=True, cache=True)
def _cast_single_ray(elevation, visible, r0, c0, z0, r1, c1, rows, cols, target_height):
    """Bresenham's Line Algorithm with continuous horizon slope tracking."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    
    err = dc - dr
    r, c = r0, c0
    max_slope = -99999.0
    
    while True:
        if 0 <= r < rows and 0 <= c < cols:
            dist = math.sqrt((r - r0)**2 + (c - c0)**2)
            if dist > 0:
                # Add human clearance offset to the target cell elevation
                dz = (elevation[r, c] + target_height) - z0
                slope = dz / dist
                if slope >= max_slope:
                    visible[r, c] = 1
                    max_slope = slope
            else:
                visible[r, c] = 1
        
        if r == r1 and c == c1:
            break
            
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr

@njit(fastmath=True, cache=True)
def _compute_point_viewshed(elevation, r0, c0, z0, target_height):
    """Traces lines of sight from a single observer position to all map edges."""
    rows, cols = elevation.shape
    visible = np.zeros((rows, cols), dtype=np.int8)
    
    if 0 <= r0 < rows and 0 <= c0 < cols:
        visible[r0, c0] = 1

    # Project rays to horizontal perimeters
    for c in range(cols):
        _cast_single_ray(elevation, visible, r0, c0, z0, 0, c, rows, cols, target_height)
        _cast_single_ray(elevation, visible, r0, c0, z0, rows - 1, c, rows, cols, target_height)
    # Project rays to vertical perimeters
    for r in range(rows):
        _cast_single_ray(elevation, visible, r0, c0, z0, r, 0, rows, cols, target_height)
        _cast_single_ray(elevation, visible, r0, c0, z0, r, cols - 1, rows, cols, target_height)
        
    return visible

@njit(fastmath=True, cache=True)
def generate_master_viewshed(elevation, observer_indices, observer_altitudes, target_height):
    """Sequentially aggregates individual viewpoint visibility matrices into a single master map."""
    rows, cols = elevation.shape
    master_map = np.zeros((rows, cols), dtype=np.int8)
    num_observers = observer_indices.shape[0]
    
    for i in range(num_observers):
        r0 = observer_indices[i, 0]
        c0 = observer_indices[i, 1]
        z0 = observer_altitudes[i]
        
        single_mask = _compute_point_viewshed(elevation, r0, c0, z0, target_height)
        
        for r in range(rows):
            for c in range(cols):
                if single_mask[r, c] == 1:
                    master_map[r, c] = 1
                    
    return master_map

# --- Hardened Spatial Data Processing Pipelines ---

def parse_and_simplify_gpx(file, epsilon=0.0001):
    """Parses track data safely into clean float arrays and runs trajectory simplification."""
    try:
        gpx = gpxpy.parse(file)
    except Exception as e:
        st.error(f"Failed to parse track asset. Ensure the file format is a valid uncorrupted GPX. Error: {str(e)}")
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                z_val = point.elevation if point.elevation is not None else np.nan
                points.append([point.latitude, point.longitude, z_val])
                
    if not points:
        st.error("No valid track geometry or coordinates found within the uploaded file.")
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        
    all_points = np.array(points, dtype=np.float64)
    
    # Process path trajectory reductions on spatial positions
    coords_2d = all_points[:, :2]
    simplified_coords = rdp(coords_2d, epsilon=epsilon)
    
    simplified_points = []
    for sc in simplified_coords:
        idx = np.argmin(np.sum((coords_2d - sc)**2, axis=1))
        simplified_points.append(all_points[idx])
        
    return all_points, np.array(simplified_points, dtype=np.float64)

def get_bounding_box(points, buffer_km):
    """Extracts exact coordinates framing the route segment including safe margin extensions."""
    lats = points[:, 0]
    lons = points[:, 1]
    avg_lat = np.nanmean(lats)
    
    lat_buffer = buffer_km / 111.32
    lon_buffer = buffer_km / (111.32 * max(math.cos(math.radians(avg_lat)), 0.1))
    
    return (float(np.nanmin(lats) - lat_buffer), float(np.nanmax(lats) + lat_buffer),
            float(np.nanmin(lons) - lon_buffer), float(np.nanmax(lons) + lon_buffer))

def download_dem(min_lat, max_lat, min_lon, max_lon, api_key):
    """Retrieves high-resolution spatial terrain models using safe cross-platform file caching."""
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        'demtype': 'SRTMGL1', 'south': min_lat, 'north': max_lat, 'west': min_lon, 'east': max_lon,
        'outputFormat': 'GTiff', 'API_Key': api_key
    }
    try:
        response = requests.get(url, params=params, stream=True, timeout=30)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", mode='wb') as temp_file:
                for chunk in response.iter_content(chunk_size=65536):
                    temp_file.write(chunk)
                return temp_file.name
        else:
            st.sidebar.error(f"Data Connection Refused ({response.status_code}): {response.text}")
    except Exception as e:
        st.sidebar.error(f"Network transport anomaly encountered: {e}")
    return None

# --- Application Controller Integration ---

with st.sidebar:
    st.header("⚙️ System Configurations")
    gpx_file = st.file_uploader("Upload Track (.gpx)", type=['gpx'])
    buffer_km = st.slider("Analysis Radius (km)", 1.0, 6.0, 2.0, 0.5)
    api_key = st.text_input("OpenTopography Token", type="password")
    
    st.markdown("---")
    epsilon = st.slider("Simplification Delta", 0.00001, 0.0005, 0.0001, step=0.00001, format="%.5f")
    eye_height = st.slider("Observer Height Offset (m)", 1.0, 3.0, 1.75, 0.25)
    analyze_btn = st.button("Initialize Pipeline", type="primary")

if analyze_btn:
    if not gpx_file or not api_key.strip():
        st.warning("Missing track properties or valid API authorization credentials.")
    else:
        full_path, simplified_path = parse_and_simplify_gpx(gpx_file, epsilon=epsilon)
        
        if len(full_path) > 0:
            dem_path = None
            try:
                min_lat, max_lat, min_lon, max_lon = get_bounding_box(full_path, buffer_km)
                
                with st.spinner("Acquiring regional terrain elevation maps..."):
                    dem_path = download_dem(min_lat, max_lat, min_lon, max_lon, api_key)
                    
                if dem_path:
                    with rasterio.open(dem_path) as src:
                        elevation = src.read(1).astype(np.float32)
                        if src.nodata is not None:
                            elevation[elevation == src.nodata] = 0.0
                        transform = src.transform
                        
                        # Downsample density constraints on exceptionally large bounds to safeguard processing
                        factor = 2 if elevation.size > 3000000 else 1
                        if factor > 1:
                            elevation = elevation[::factor, ::factor]

                        # Generate structured 1D spatial mapping paths safely before combining via meshgrid
                        xs_coords, _ = rasterio.transform.xy(transform, [0]*elevation.shape[1], np.arange(elevation.shape[1]) * factor)
                        _, ys_coords = rasterio.transform.xy(transform, np.arange(elevation.shape[0]) * factor, [0]*elevation.shape[0])
                        xs, ys = np.meshgrid(xs_coords, ys_coords)
                        
                        # Initialize position nodes for validation calculations
                        obs_indices = []
                        obs_altitudes = []
                        
                        for pt in simplified_path:
                            r, c = rowcol(transform, pt[1], pt[0])
                            r, c = int(r // factor), int(c // factor)
                            
                            if 0 <= r < elevation.shape[0] and 0 <= c < elevation.shape[1]:
                                obs_indices.append([r, c])
                                ground_z = float(elevation[r, c])
                                
                                # Evaluate elevation parameters to catch anomalies or flat values
                                if np.isnan(pt[2]) or pt[2] <= 0.0:
                                    obs_z = ground_z + eye_height
                                else:
                                    obs_z = max(float(pt[2]), ground_z) + eye_height
                                obs_altitudes.append(obs_z)
                        
                        with st.spinner("Executing optimized ray casting loops..."):
                            if obs_indices:
                                cumulative_mask = generate_master_viewshed(
                                    elevation, 
                                    np.array(obs_indices, dtype=np.int32), 
                                    np.array(obs_altitudes, dtype=np.float32),
                                    eye_height
                                )
                            else:
                                cumulative_mask = np.zeros(elevation.shape, dtype=np.int8)

                    # --- Visual Render Frameworks ---
                    tab1, tab2 = st.tabs(["📊 Interactive 3D WebGL", "🗺️ 2D Overlay Layout"])
                    
                    with tab1:
                        with st.spinner("Formatting 3D WebGL space graphics canvas..."):
                            max_dim = max(elevation.shape)
                            viz_step = math.ceil(max_dim / 450) if max_dim > 450 else 1
                            
                            fig = go.Figure(data=[go.Surface(
                                x=xs[::viz_step, ::viz_step], 
                                y=ys[::viz_step, ::viz_step], 
                                z=elevation[::viz_step, ::viz_step],
                                surfacecolor=cumulative_mask[::viz_step, ::viz_step],
                                colorscale=[[0, 'rgba(235, 64, 52, 0.45)'], [1, 'rgba(52, 235, 113, 0.5)']],
                                cmin=0, cmax=1, showscale=False,
                                lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.4, specular=0.1)
                            )])
                            
                            plot_lats = full_path[:, 0]
                            plot_lons = full_path[:, 1]
                            
                            # Fallback projection checks for trace-line path visibility
                            mean_elev = float(np.nanmean(elevation))
                            plot_zs = [float(pt[2]) + 12 if not np.isnan(pt[2]) else mean_elev + 12 for pt in full_path]
                            
                            fig.add_trace(go.Scatter3d(
                                x=plot_lons, y=plot_lats, z=plot_zs,
                                mode='lines', line=dict(color='rgb(37, 99, 235)', width=6), name='Track Line'
                            ))
                            fig.update_layout(
                                autosize=True, scene=dict(aspectratio=dict(x=1, y=1, z=0.22)),
                                margin=dict(l=0, r=0, b=0, t=0)
                            )
                            st.plotly_chart(fig, width="stretch")
                            

                            # --- 100% Offline Standalone 3D HTML Export ---
                            # Extract the HTML string directly, bypassing the byte buffer
                            plotly_html_str = fig.to_html(include_plotlyjs=True, full_html=True)

                            st.download_button(
                                label="📥 Export Standalone 3D WebGL Scene (Zero Dependencies)",
                                data=plotly_html_str,
                                file_name="viewshed_analysis_3d.html",
                                mime="text/html",
                                key="download_plotly_html"
)
                            
                    with tab2:
                        with st.spinner("Compiling structural 2D geographic maps..."):
                            center_lat = float(np.nanmean(plot_lats))
                            center_lon = float(np.nanmean(plot_lons))
                            m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
                            
                            path_coords = list(zip(plot_lats, plot_lons))
                            folium.PolyLine(path_coords, color="#2563eb", weight=4, opacity=0.95).add_to(m)
                            
                            for i, pt in enumerate(simplified_path):
                                folium.CircleMarker(
                                    location=[pt[0], pt[1]], radius=4, color="#ef4444", fill=True,
                                    popup=f"Waypoint Verification Node #{i+1}"
                                ).add_to(m)
                                
                            st_folium(m, width=1200, height=600, returned_objects=[])
                            
                            # --- Full-Template 2D Map HTML Export ---
                            # --- Full-Template 2D Map HTML Export ---
                            # Extract the full HTML template string directly from the map root
                            folium_html_str = m.get_root().render()

                            st.download_button(
                                label="📥 Export Standalone 2D Map Asset",
                                data=folium_html_str,
                                file_name="viewshed_map_2d.html",
                                mime="text/html",
                                key="download_folium_html"
                            )
                            
            finally:
                # Guarantee memory space isolation by cleaning temporary assets safely
                if dem_path and os.path.exists(dem_path):
                    try:
                        os.remove(dem_path)
                    except Exception:
                        pass
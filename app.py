import numpy as np
import pandas as pd
import streamlit as st
import folium
import plotly.express as px
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from math import radians, sin, cos, asin, sqrt

st.set_page_config(page_title="Dark Store Location Optimizer", page_icon="📍", layout="wide")
st.title("📍 Dark Store Location Optimization")
st.caption("Profit-maximizing store selection with a maximum customer-to-store distance constraint.")

# Chennai study-area bounding box; points are synthetic unless replaced with real coordinates.
CHENNAI_BOUNDS = {"lat_min": 12.85, "lat_max": 13.20, "lon_min": 80.12, "lon_max": 80.32}

def haversine_matrix(customers, stores):
    """Return customer-store great-circle distances in km."""
    lat1 = np.radians(customers[:, 0])[:, None]
    lon1 = np.radians(customers[:, 1])[:, None]
    lat2 = np.radians(stores[:, 0])[None, :]
    lon2 = np.radians(stores[:, 1])[None, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def generate_data(n_customers, n_stores, seed=42):
    """Generate reproducible synthetic points inside a Chennai-area bounding box."""
    rng = np.random.default_rng(seed)
    # Mixture of clusters to look more city-like than a uniform scatter.
    centers = np.array([[13.0827, 80.2707], [13.0569, 80.2425],
                        [13.0358, 80.2101], [13.1067, 80.2206],
                        [12.9716, 80.2209], [13.1250, 80.2900]])
    weights = np.array([.28, .18, .14, .14, .14, .12])
    chosen = rng.choice(len(centers), size=n_customers, p=weights)
    customer_points = centers[chosen] + rng.normal(0, [0.025, 0.027], size=(n_customers, 2))
    customer_points[:, 0] = np.clip(customer_points[:, 0], CHENNAI_BOUNDS["lat_min"], CHENNAI_BOUNDS["lat_max"])
    customer_points[:, 1] = np.clip(customer_points[:, 1], CHENNAI_BOUNDS["lon_min"], CHENNAI_BOUNDS["lon_max"])
    store_points = np.column_stack([
        rng.uniform(CHENNAI_BOUNDS["lat_min"], CHENNAI_BOUNDS["lat_max"], n_stores),
        rng.uniform(CHENNAI_BOUNDS["lon_min"], CHENNAI_BOUNDS["lon_max"], n_stores)
    ])
    return customer_points, store_points

def solve_location_model(distances, max_distance_km, max_open, revenue, delivery_cost_per_km, fixed_cost):
    """
    MILP variables: x_j (store open) and y_ij (customer i assigned to store j).
    Only distance-feasible customer/store pairs are included.
    """
    n, m = distances.shape
    feasible_i, feasible_j = np.where(distances <= max_distance_km + 1e-9)
    p = len(feasible_i)
    if p == 0:
        return {"success": True, "status": "Optimal (no feasible assignments)", "profit": 0.0,
                "opened": [], "assigned_store": np.full(n, -1), "served": 0,
                "revenue": 0.0, "delivery_cost": 0.0, "fixed_cost": 0.0,
                "avg_distance": 0.0, "max_assigned_distance": 0.0, "solver_message": "No pairs within distance limit."}

    # Variable order: x[0:m], then y[0:p].
    nvars = m + p
    objective = np.zeros(nvars)
    objective[:m] = fixed_cost
    objective[m:] = -(revenue - delivery_cost_per_km * distances[feasible_i, feasible_j])

    # Rows: one per customer (at most one assignment), one per feasible pair (y_ij <= x_j),
    # and one store-count row.
    nrows = n + p + 1
    A = lil_matrix((nrows, nvars), dtype=float)
    lb = np.full(nrows, -np.inf)
    ub = np.full(nrows, np.inf)

    # Customer assignment: sum_j y_ij <= 1
    for k, i in enumerate(feasible_i):
        A[i, m + k] = 1.0
    ub[:n] = 1.0

    # Assignment can only be to an open store: y_ij - x_j <= 0
    for k, j in enumerate(feasible_j):
        row = n + k
        A[row, j] = -1.0
        A[row, m + k] = 1.0
        ub[row] = 0.0

    # At most max_open stores.
    A[n + p, :m] = 1.0
    ub[n + p] = min(max_open, m)

    integrality = np.ones(nvars, dtype=np.int8)
    bounds = Bounds(np.zeros(nvars), np.ones(nvars))
    result = milp(c=objective, integrality=integrality, bounds=bounds,
                  constraints=LinearConstraint(csr_matrix(A), lb, ub),
                  options={"time_limit": 120, "mip_rel_gap": 0.02})

    if result.x is None:
        return {"success": False, "status": "No feasible solution returned", "profit": np.nan,
                "opened": [], "assigned_store": np.full(n, -1), "served": 0,
                "revenue": 0.0, "delivery_cost": 0.0, "fixed_cost": 0.0,
                "avg_distance": 0.0, "max_assigned_distance": 0.0,
                "solver_message": result.message}

    x = result.x[:m] > 0.5
    y = result.x[m:] > 0.5
    assigned = np.full(n, -1, dtype=int)
    for k in np.where(y)[0]:
        assigned[feasible_i[k]] = feasible_j[k]
    served_idx = np.where(assigned >= 0)[0]
    served = len(served_idx)
    chosen_distances = distances[served_idx, assigned[served_idx]] if served else np.array([])
    total_revenue = served * revenue
    total_delivery = float(np.sum(chosen_distances) * delivery_cost_per_km)
    total_fixed = int(np.sum(x)) * fixed_cost
    profit = total_revenue - total_delivery - total_fixed
    status = "Optimal" if result.status == 0 else f"Feasible incumbent — {result.message}"
    return {"success": True, "status": status, "profit": profit,
            "opened": list(np.where(x)[0]), "assigned_store": assigned, "served": served,
            "revenue": total_revenue, "delivery_cost": total_delivery, "fixed_cost": total_fixed,
            "avg_distance": float(np.mean(chosen_distances)) if served else 0.0,
            "max_assigned_distance": float(np.max(chosen_distances)) if served else 0.0,
            "solver_message": result.message}

def make_map(customers, stores, result, max_distance_km):
    center = [float(np.mean(customers[:, 0])), float(np.mean(customers[:, 1]))]
    fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    opened = set(result["opened"])
    # Candidate sites: selected stores are green, unopened candidates are blue.
    for j, (lat, lon) in enumerate(stores):
        is_open = j in opened
        folium.Marker(
            [float(lat), float(lon)],
            tooltip=f"{'OPEN' if is_open else 'Candidate'} store {j+1}",
            popup=f"Store {j+1}<br>Status: {'Opened' if is_open else 'Not opened'}",
            icon=folium.Icon(color="green" if is_open else "blue", icon="home", prefix="fa")
        ).add_to(fmap)
        if is_open:
            folium.Circle([float(lat), float(lon)], radius=max_distance_km * 1000,
                          color="green", weight=1, fill=True, fill_opacity=0.06,
                          tooltip=f"{max_distance_km:g} km service radius").add_to(fmap)

    # Cluster customer markers. For large datasets, show a representative subset for map speed.
    cluster = MarkerCluster(name="Customers (sample)").add_to(fmap)
    n = len(customers)
    sample_ids = np.arange(n) if n <= 2500 else np.linspace(0, n - 1, 2500, dtype=int)
    assigned = result["assigned_store"]
    for i in sample_ids:
        lat, lon = customers[i]
        j = assigned[i]
        color = "green" if j >= 0 else "red"
        folium.CircleMarker([float(lat), float(lon)], radius=2, color=color,
                            fill=True, fill_opacity=0.55,
                            tooltip=f"Customer {i+1}: {'served by store '+str(j+1) if j >= 0 else 'unserved'}").add_to(cluster)

    # Draw a limited number of assignment lines to avoid a very heavy map.
    served_ids = np.where(assigned >= 0)[0]
    if len(served_ids) > 300:
        served_ids = served_ids[np.linspace(0, len(served_ids)-1, 300, dtype=int)]
    for i in served_ids:
        j = assigned[i]
        folium.PolyLine([customers[i].tolist(), stores[j].tolist()],
                        color="#228B22", weight=1, opacity=0.25).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap

with st.sidebar:
    st.header("Model inputs")
    n_stores = st.number_input("Candidate dark store locations", min_value=2, max_value=100, value=50, step=1)
    n_customers = st.number_input("Number of customers", min_value=10, max_value=10000, value=1000, step=100)
    max_distance = st.number_input("Maximum customer-to-store distance (km)", min_value=0.1, max_value=30.0, value=3.0, step=0.5)
    max_open = st.number_input("Maximum stores allowed to open", min_value=1, max_value=int(n_stores), value=min(5, int(n_stores)), step=1)
    st.subheader("Illustrative financial assumptions")
    revenue = st.number_input("Contribution per served customer (₹)", min_value=0.0, value=100.0, step=10.0)
    delivery_cost = st.number_input("Delivery cost per customer per km (₹)", min_value=0.0, value=5.0, step=1.0)
    fixed_cost = st.number_input("Fixed cost per open store (₹)", min_value=0.0, value=20000.0, step=5000.0)
    seed = st.number_input("Random seed (reproducibility)", min_value=0, max_value=999999, value=42, step=1)
    run = st.button("Run optimization", type="primary", use_container_width=True)

st.info("Geography is simulated within a Chennai-area bounding box. These are not verified customer addresses or real commercial properties.")

if run or "main_result" not in st.session_state:
    with st.spinner("Generating locations and solving the MILP. Large instances may take a few minutes..."):
        customers, stores = generate_data(int(n_customers), int(n_stores), int(seed))
        distance_matrix = haversine_matrix(customers, stores)
        main_result = solve_location_model(distance_matrix, float(max_distance), int(max_open),
                                           float(revenue), float(delivery_cost), float(fixed_cost))
        st.session_state["customers"] = customers
        st.session_state["stores"] = stores
        st.session_state["distance_matrix"] = distance_matrix
        st.session_state["main_result"] = main_result
        st.session_state["main_inputs"] = (float(max_distance), float(revenue), float(delivery_cost), float(fixed_cost))

if "main_result" in st.session_state:
    customers = st.session_state["customers"]
    stores = st.session_state["stores"]
    distance_matrix = st.session_state["distance_matrix"]
    result = st.session_state["main_result"]
    used_distance, used_revenue, used_delivery, used_fixed = st.session_state["main_inputs"]

    st.subheader("Optimization result")
    st.write(f"**Solver status:** {result['status']}")
    if result["success"]:
        total = len(customers)
        served = result["served"]
        kpi = pd.DataFrame({
            "KPI": ["Candidate locations", "Total customers", "Maximum distance (km)", "Stores opened",
                    "Customers served", "Customers unserved", "Customer coverage (%)",
                    "Revenue (₹)", "Delivery cost (₹)", "Fixed store cost (₹)", "Total profit (₹)",
                    "Average assigned distance (km)", "Maximum assigned distance (km)"],
            "Value": [len(stores), total, used_distance, len(result["opened"]), served, total-served,
                      round(100*served/total, 2), round(result["revenue"], 2), round(result["delivery_cost"], 2),
                      round(result["fixed_cost"], 2), round(result["profit"], 2),
                      round(result["avg_distance"], 3), round(result["max_assigned_distance"], 3)]
        })
        st.dataframe(kpi, use_container_width=True, hide_index=True)
        st.caption("The maximum assigned distance is calculated only for served customers and should not exceed the chosen limit.")

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Detailed solution map")
            st_folium(make_map(customers, stores, result, used_distance), use_container_width=True, height=600)
            st.caption("Green store markers = selected stores; blue = unselected candidates; green customers = served; red customers = unserved. Customer markers and assignment lines are sampled for map performance.")
        with right:
            st.subheader("Opened locations")
            if result["opened"]:
                opened_df = pd.DataFrame([{"Store": j+1, "Latitude": stores[j,0], "Longitude": stores[j,1],
                                           "Customers assigned": int(np.sum(result["assigned_store"] == j))}
                                          for j in result["opened"]])
                st.dataframe(opened_df, use_container_width=True, hide_index=True)
                st.download_button("Download opened stores CSV", opened_df.to_csv(index=False).encode("utf-8"),
                                   "opened_dark_stores.csv", "text/csv")
            else:
                st.write("No stores were opened under the current assumptions.")

        st.divider()
        st.subheader("Required experiment: profit versus maximum number of stores")
        st.write("Runs five additional optimizations for 10,000 customers and 50 candidate locations, using the same distance and financial assumptions.")
        if st.button("Run 2, 5, 10, 20, 25-store comparison", type="primary"):
            with st.spinner("Solving five scenarios. This can take several minutes..."):
                exp_customers, exp_stores = generate_data(10000, 50, int(seed))
                exp_distances = haversine_matrix(exp_customers, exp_stores)
                rows = []
                progress = st.progress(0)
                limits = [2, 5, 10, 20, 25]
                for idx, limit in enumerate(limits):
                    rr = solve_location_model(exp_distances, used_distance, limit,
                                              used_revenue, used_delivery, used_fixed)
                    rows.append({"Maximum stores allowed": limit,
                                 "Stores opened": len(rr["opened"]),
                                 "Customers served": rr["served"],
                                 "Profit (₹)": rr["profit"],
                                 "Solver status": rr["status"]})
                    progress.progress((idx+1)/len(limits))
                exp_df = pd.DataFrame(rows)
                st.session_state["experiment_df"] = exp_df

        if "experiment_df" in st.session_state:
            exp_df = st.session_state["experiment_df"]
            st.dataframe(exp_df, use_container_width=True, hide_index=True)
            fig = px.line(exp_df, x="Maximum stores allowed", y="Profit (₹)", markers=True,
                          title="Optimized profit for each store limit")
            fig.update_layout(xaxis=dict(tickmode="array", tickvals=[2,5,10,20,25]))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download experiment results CSV", exp_df.to_csv(index=False).encode("utf-8"),
                               "profit_comparison.csv", "text/csv")
    else:
        st.error("The solver did not return a feasible incumbent. Try reducing the customer count or changing the inputs.")
        st.write(result["solver_message"])

st.divider()
st.caption("Model note: binary MILP with at-most-one customer assignment, open-store linking, store-count limit, and distance-feasible assignments only. A time-limited solver result may be a feasible incumbent rather than a proven global optimum.")

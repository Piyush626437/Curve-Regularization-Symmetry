import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import leastsq
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
# from bezier import Curve
# import svgwrite
# import cairosvg

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

def fit_lines(paths_XYs):
    lines = []
    for path in paths_XYs:
        for XY in path:
            X = XY[:, 0].reshape(-1, 1)
            y = XY[:, 1]
            model = LinearRegression().fit(X, y)
            lines.append((model.coef_[0], model.intercept_))
    return lines

def fit_circle(x, y):
    def residuals(params):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    x_m, y_m = np.mean(x), np.mean(y)
    r_init = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))
    initial_guess = (x_m, y_m, r_init)
    result, _ = leastsq(residuals, initial_guess)
    return result

def detect_circles(paths_XYs):
    circles = []
    for path in paths_XYs:
        for XY in path:
            x, y = XY[:, 0], XY[:, 1]
            params = fit_circle(x, y)
            circles.append(params)
    return circles

def detect_rectangles(paths_XYs):
    rectangles = []
    for path in paths_XYs:
        for XY in path:
            points = np.array(XY, dtype=np.float32)
            epsilon = 0.02 * cv2.arcLength(points, True)
            approx = cv2.approxPolyDP(points, epsilon, True)
            if len(approx) == 4:
                rectangles.append(approx)
    return rectangles

def is_regular_polygon(points):
    distances = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
    distances = np.array(distances)
    return np.allclose(np.std(distances), 0)

def detect_polygons(paths_XYs):
    polygons = []
    for path in paths_XYs:
        for XY in path:
            if is_regular_polygon(XY):
                polygons.append(XY)
    return polygons

def detect_stars(paths_XYs):
    stars = []
    for path in paths_XYs:
        for XY in path:
            # Placeholder for star detection logic
            stars.append(XY)
    return stars

def check_reflection_symmetry(points):
    # Placeholder for actual symmetry check
    return True

def fit_bezier_curve(points):
    curve = Curve.from_nodes(points)
    return curve

def create_bezier_curves(paths_XYs):
    bezier_curves = []
    for path in paths_XYs:
        for XY in path:
            bezier_curve = fit_bezier_curve(XY)
            bezier_curves.append(bezier_curve)
    return bezier_curves

def complete_curve(points):
    x = points[:, 0]
    y = points[:, 1]
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(min(x), max(x), num=100)
    y_new = interp_func(x_new)
    return np.column_stack((x_new, y_new))

def complete_curves(paths_XYs):
    completed_curves = []
    for path in paths_XYs:
        for XY in path:
            completed_curve = complete_curve(XY)
            completed_curves.append(completed_curve)
    return completed_curves

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    dwg.add(group)
    dwg.save()
    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W, output_height=fact * H, background_color='white')

def process_shapes(paths_XYs):
    lines = fit_lines(paths_XYs)
    circles = detect_circles(paths_XYs)
    rectangles = detect_rectangles(paths_XYs)
    polygons = detect_polygons(paths_XYs)
    stars = detect_stars(paths_XYs)
    symmetrical_curves = [check_reflection_symmetry(XY) for path in paths_XYs for XY in path]
    bezier_curves = create_bezier_curves(paths_XYs)
    completed_curves = complete_curves(paths_XYs)

    return {
        'lines': lines,
        'circles': circles,
        'rectangles': rectangles,
        'polygons': polygons,
        'stars': stars,
        'bezier_curves': bezier_curves,
        'completed_curves': completed_curves
    }

def main():
    input_path = 'path_to_input.csv'  # Replace with your input CSV path
    paths_XYs = read_csv(input_path)
    results = process_shapes(paths_XYs)
    plot(paths_XYs)  # Original input visualization
    polylines2svg(paths_XYs, 'output.svg')  # Save SVG for further use

if __name__ == "__main__":
    main()

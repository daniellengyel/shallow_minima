import matplotlib.pyplot as plt
import autograd.numpy as np
import time, os, subprocess

from scipy import stats

import pickle
import yaml

def create_animation_pictures(path, X, Y, Z, graph_type="contour", graph_details={}):
    """
    path: path[:, 0]=path_x, path[:, 1]=path_y, path[:, 2]=path_z"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/{}".format(time.time())
    os.mkdir(ani_path)

    for i in range(len(path)):
        fig, ax = plt.subplots()
        if graph_type == "contour":
            ax.contour(X, Y, Z, graph_details["lines"])
        else:
            ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0][0], X[0][-1], Y[-1][0], Y[0][0]], interpolation=graph_details["interpolation"])

        # plot the path
        for j in range(max(0, i - 20), i):
            ax.plot(path[j - 1:j + 1, 0], path[j - 1:j + 1, 1], "--*", color="red", alpha=np.exp(-(i - j - 1) / 5.))

        plt.savefig(ani_path + "/{}.png".format(i))
    return ani_path

# def create_animation_1d_pictures(path, X, Y):


def create_overlay_animation_pictures(path, X, Y, Z, Z1, graph_type="contour", graph_details={}):
    """
    path: path[:, 0]=path_x, path[:, 1]=path_y, path[:, 2]=path_z"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/{}".format(time.time())
    os.mkdir(ani_path)

    for i in range(len(path)):
        fig, ax = plt.subplots()
        if graph_type == "contour":
            ax.contour(X, Y, Z, graph_details["lines"])
            ax.contour(X, Y, Z1, graph_details["lines"])
        else:
            ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0][0], X[0][-1], Y[-1][0], Y[0][0]], interpolation=graph_details["interpolation"])
            ax.imshow(Z1, cmap=plt.cm.gist_earth_r, extent=[X[0][0], X[0][-1], Y[-1][0], Y[0][0]], interpolation=graph_details["interpolation"])

        # plot the path
        for j in range(max(0, i - 20), i):
            ax.plot(path[j - 1:j + 1, 0], path[j - 1:j + 1, 1], "--*", color="red", alpha=np.exp(-(i - j - 1) / 5.))

        plt.savefig(ani_path + "/{}.png".format(i))
    return ani_path


# ffmpeg -r 20 -f image2 -s 1920x1080 -i %d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
def create_animation(image_folder, video_name, screen_resolution="1920x1080", framerate=30, qaulity=25,
                     extension=".png"):
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-r", str(framerate),
            "-f", "image2",
            "-s", screen_resolution,
            "-i", os.path.join(image_folder, "%d" + extension),
            "-vcodec", "libx264",
            "-crf", str(qaulity),
            "-pix_fmt", "yuv420p",
            os.path.join(image_folder, video_name)
        ])


# NOT FULLY TESTED YET
def create_animation_density_pictures(paths):
    """
    Assume all paths have same length"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/density_{}".format(time.time())
    os.mkdir(ani_path)

    for i in range(len(paths[0])):
        X = paths[:, i, 0].T[0]
        Y = paths[:, i, 1].T[0]

        kernel = stats.gaussian_kde(np.vstack([X, Y]))
        x_min, x_max = max(-15, min(X)), min(15, max(X))
        y_min, y_max = max(-15, min(Y)), min(15, max(Y))
        positions = np.mgrid[x_min:x_max:0.2, y_min:y_max:0.2]

        Z = np.reshape(kernel(np.vstack([positions[0].ravel(), positions[1].ravel()])).T, positions[0].shape)

        fig, ax = plt.subplots()
        ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])

        plt.savefig(ani_path + "/{}.png".format(i))
    return ani_path

def create_animation_density_particles_pictures(paths):
    """
    Assume all paths have same length"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/density_{}".format(time.time())
    os.mkdir(ani_path)

    for i in range(len(paths[0])):
        fig, ax = plt.subplots()
        ax.set_xlim(-35, 35)
        ax.set_ylim(-35, 35)
        ax.plot(paths[:, i, 0, :], paths[:, i, 1, :], "o")

        plt.savefig(ani_path + "/{}.png".format(i))
    return ani_path


def create_animation_interactive_diffusion(particles_paths, second_path, X, Y, Z, ratio, second_particle_type="particles", graph_type="heatmap", graph_details=None):
    """
    Assume all paths have same length. And that the length of first and second path are off by an integer factor."""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/interactive_{}".format(time.time())
    os.mkdir(ani_path)

    trail_length = 20

    # ratio = (particles_paths.shape[1] - 1) / (second_path.shape[0] - 1)
    # assert int(ratio) == ratio

    for i in range(len(particles_paths[0])):
        p_X = particles_paths[:, i, 0].T
        p_Y = particles_paths[:, i, 1].T

        fig, ax = plt.subplots()

        if graph_type == "contour":
            ax.contour(X, Y, Z, graph_details["lines"])
        else:
            ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0][0], X[0][-1], Y[-1][0], Y[0][0]],
                      interpolation=graph_details["interpolation"])

        trail_start = int(max(0, i / ratio - trail_length))
        trail_end = int(max(0, i / ratio + 1))

        if second_particle_type == "particles":
            ax.plot(p_X, p_Y, "o", color="orange")
        else:
            kernel = stats.gaussian_kde(np.vstack([p_X, p_Y]))
            x_min, x_max = -10, 10  # max(-15, min(min(X), min(second_path[trail_start:trail_end, 0]))), min(15, max(max(X), max(second_path[trail_start:trail_end, 0])))
            y_min, y_max = -10, 10  # max(-15, min(min(Y), min(second_path[trail_start:trail_end, 1]))), min(15, max(max(Y), max(second_path[trail_start:trail_end, 1])))

            positions = np.mgrid[x_min:x_max:0.2, y_min:y_max:0.2]

            Z = np.reshape(kernel(np.vstack([positions[0].ravel(), positions[1].ravel()])).T, positions[0].shape)
            ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])


        # plot the second path
        for j in range(trail_start, trail_end):
            ax.plot(second_path[j - 1:j + 1, 0], second_path[j - 1:j + 1, 1], "--*", color="red",
                    alpha=np.exp(-(trail_end - j - 1) / 5.))

        plt.savefig(ani_path + "/{}.png".format(i))
    return ani_path


def create_end_points_pictures(end_points, X, Y, Z, graph_type="contour", graph_details={}):
    """
    path: path[:, 0]=path_x, path[:, 1]=path_y, path[:, 2]=path_z"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/end_points_{}".format(time.time())
    os.mkdir(ani_path)

    fig, ax = plt.subplots()
    if graph_type == "contour":
        ax.contour(X, Y, Z, graph_details["lines"])
    else:
        ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0][0], X[0][-1], Y[-1][0], Y[0][0]],
                  interpolation=graph_details["interpolation"])

    # plot the path
    for p in end_points:
        ax.plot(p[0], p[1], "o", color="red")

    plt.savefig(ani_path + "/end_points.png")

    return ani_path

def show_end_points_pictures(end_points, X, Y, Z, graph_type="contour", graph_details={}):
    """
    path: path[:, 0]=path_x, path[:, 1]=path_y, path[:, 2]=path_z"""


    fig, ax = plt.subplots()
    if graph_type == "contour":
        ax.contour(X, Y, Z, graph_details["lines"])
    else:
        ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0][0], X[0][-1], Y[-1][0], Y[0][0]],
                  interpolation=graph_details["interpolation"])

    # plot the path
    for p in end_points:
        ax.plot(p[0], p[1], "o", color="red")

    plt.show()


def save_analytics_paths(analytics, paths, name=""):
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    path = "./tmp/analytics_paths_{}_{}".format(name, time.time())
    os.mkdir(path)

    with open(path + "/analytics.pkl", "wb") as f:
        pickle.dump(analytics, f)

    with open(path + "/paths.pkl", "wb") as f:
        pickle.dump(paths, f)


def load_analytics_paths(path):
    with open(path + "/analytics.pkl", "rb") as f:
        analytics = pickle.load(f)

    with open(path + "/paths.pkl", "rb") as f:
        paths = pickle.load(f)

    return analytics, paths


def create_animation_1d_pictures_particles(all_paths, X, Y, folder_name="", graph_details={"p_size": 1, "density_function": None}):
    """path: path[:, 0]=path_x, path[:, 1]=path_y"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/1d_{0}_{1}".format(folder_name, time.strftime("%Y%m%d-%H%M%S"))
    os.mkdir(ani_path)

    available_colors = ["red", "green"]

    num_tau, num_particles, tau, dim = all_paths.shape

    density_function = graph_details["density_function"]



    for i in range(len(all_paths)):
        curr_paths = all_paths[i]

        color_use = available_colors[i % len(available_colors)]

        for j in range(tau):

            if density_function is not None:
                fig = plt.figure(figsize=(14, 10))
                gs = fig.add_gridspec(4, 1)
                ax = fig.add_subplot(gs[1:4, :])
                ax2 = fig.add_subplot(gs[0, :])
            else:
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(1, 1, 1)

            if density_function is not None:
                Y_density = density_function(X, curr_paths[:, j, 0])
                ax2.plot(X, Y_density)

            fig.suptitle(folder_name, fontsize=20)

            ax.plot(X, Y)
            ax.plot(curr_paths[:, j, 0], curr_paths[:, j, 1], "o", color=color_use, markersize=graph_details["p_size"])

            plt.savefig(ani_path + "/{}.png".format(i * tau + j))

            plt.close()


    return ani_path


def create_animation_2d_pictures_particles(all_paths, X, Y, Z, folder_name="", graph_details={"p_size": 1, "density_function": None}):
    """path: path[:, 0]=path_x, path[:, 1]=path_y, path[:, 2] = path_z"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    if not os.path.isdir("./tmp/2D"):
        os.mkdir("./tmp/2D")
    ani_path = "./tmp/2D/2d_{0}_{1}".format(folder_name, time.strftime("%Y%m%d-%H%M%S"))
    os.mkdir(ani_path)

    available_colors = ["red", "green"]

    num_tau, num_particles, tau, dim = all_paths.shape

    density_function = graph_details["density_function"]

    for i in range(len(all_paths)):
        curr_paths = all_paths[i]

        color_use = available_colors[i % len(available_colors)]

        for j in range(tau):

            if density_function is not None:
                fig = plt.figure(figsize=(30, 10))
                gs = fig.add_gridspec(1, 4)
                ax = fig.add_subplot(gs[:, 0:2])
                ax2 = fig.add_subplot(gs[:, 2:4])
            else:
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(1, 1, 1)

            if density_function is not None:
                inp = np.array(np.meshgrid(X, Y)).reshape(2, len(X) * len(Y))
                Z_density = density_function(inp, curr_paths[:, j, :2]).reshape(len(X), len(Y))

                if graph_details["type"] == "contour":
                    ax2.contour(X, Y, Z_density, 40)
                else:
                    ax2.imshow(Z_density, cmap=plt.cm.gist_earth_r, extent=[X[0], X[-1], Y[-1], Y[0]],
                              interpolation=graph_details["interpolation"])
            else:
                ax.plot(curr_paths[:, j, 0], curr_paths[:, j, 1], "o", color=color_use, markersize=graph_details["p_size"])

            fig.suptitle(folder_name, fontsize=20)

            if graph_details["type"] == "contour":
                ax.contour(X, Y, Z, 40)
            else:
                ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0], X[-1], Y[-1], Y[0]],
                          interpolation=graph_details["interpolation"])


            plt.savefig(ani_path + "/{}.png".format(i * tau + j))

            plt.close()

    return ani_path


def create_animation_2d_pictures_particles_sampling(all_paths, X, Y, Z, folder_name="", graph_type="contour", graph_details={}):
    """path: path[:, 0]=path_x, path[:, 1]=path_y"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/2d_particles_{0}_{1}".format(folder_name, time.time())
    os.mkdir(ani_path)

    available_colors = ["red", "green"]

    for i in range(len(all_paths)):
        curr_paths = all_paths[i]

        color_use = available_colors[i % len(available_colors)]

        for j in range(len(curr_paths)):
            fig, ax = plt.subplots()

            if graph_type == "contour":
                ax.contour(X, Y, Z, graph_details["lines"])
            else:
                ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0], X[-1], Y[-1], Y[0]],
                          interpolation=graph_details["interpolation"])

            ax.plot(curr_paths[:, j, 0], curr_paths[:, j, 1], "o", color=color_use)

            plt.savefig(ani_path + "/{}.png".format(i * len(curr_paths) + j))

            plt.close(fig)

    return ani_path

def create_animation_2d_pictures_particles_interaction(all_paths, X, Y, Z, folder_name="", graph_type="contour", graph_details={}):
    """path: path[:, 0]=path_x, path[:, 1]=path_y"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/2d_particles_{0}_{1}".format(folder_name, time.time())
    os.mkdir(ani_path)


    for j in range(len(all_paths[0])):

        fig, ax = plt.subplots()

        if graph_type == "contour":
            ax.contour(X, Y, Z, graph_details["lines"])
        else:
            ax.imshow(Z, cmap=plt.cm.gist_earth_r, extent=[X[0][0], X[0][-1], Y[-1][0], Y[0][0]],
                      interpolation=graph_details["interpolation"])

        ax.plot(all_paths[:, j, 0], all_paths[:, j, 1], "o", color="orange")

        plt.savefig(ani_path + "/{}.png".format(j))

        plt.close(fig)

    return ani_path


def remove_png(dir_path):
    files = os.listdir(dir_path)
    for item in files:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_path, item))


def save_config(dir_path, config):
    with open(dir_path + "/process.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)






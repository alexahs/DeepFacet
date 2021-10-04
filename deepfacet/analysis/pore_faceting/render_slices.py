from ovito.modifiers import *
from ovito.vis import *
from ovito.io import *
import numpy as np
import math
from dataclasses import dataclass
from pathlib import Path
# from PySide2 import QtCore
# from PySide2.QtWidgets import QApplication



def render_slice(infile, outfile_prefix, frames, orientation=(1,0,0), particle_radius = 1.2, fig_size=(800,600), background=(0,0,0), alpha=1):
    """
    Renders images of a systems cross section
    """


    if not hasattr(frames, '__len__'):
        frames = [frames]

    pipeline = import_file(infile)
    num_frames = pipeline.source.num_frames
    assert(num_frames-1 >= max(frames)), f"only {num_frames} frames in pipeline ({max(frames)} >= {num_frames})"

    size = pipeline.source.data.cell[0,0]

    sqrt2 = np.sqrt(2)
    if orientation == (1, 0, 0):
        orient = "100"
        # d_min = size/2 - 2
        # d_max = size/2 + 2
        d_min = size/2
    elif orientation == (1, 1, 0):
        orient = "110"
        d_min = size*np.sqrt(2)/2 - 2
        print(d_min)
        # exit()
        # d_min = size*np.sqrt(2)/2 - 3
        # d_max = size*np.sqrt(2)/2 + 3
        # w = round(fig_size[0]*np.sqrt(2))
        # h = round(fig_size[1]*np.sqrt(2))
        # fig_size = (w, h)
    else:
        raise NotImplementedError

    # print(fig_size)

    pipeline.modifiers.append(IdentifyDiamondModifier())
    pipeline.modifiers.append(SliceModifier(normal=orientation, distance=d_min, inverse=False))
    # pipeline.modifiers.append(SliceModifier(normal=orientation, distance=d_max, inverse=False))
    pipeline.modifiers.append(AmbientOcclusionModifier(intensity=0.6))

    data = pipeline.compute()
    data.particles.vis.radius = particle_radius

    """
    (110):
    pos xyz = [n, n, n]/2
    dir xyz = [0.7, 0.7, 0]

    (100):
    pos xyz = [n, n, n]/2
    dir xyz = [1, 0, 0]
    """


    pipeline.add_to_scene()
    # if system_size is None:
    vp = Viewport(type = Viewport.Type.Ortho, camera_dir=orientation)
    # else:
    c_pos = tuple([size/2]*3)
    if orientation == (1, 1, 0):
        c_dir = (-0.7, -0.7, 0)
    elif orientation == (1, 0, 0):
        c_dir = (-1, 0, 0)
    # vp = Viewport(type = Viewport.Type.Ortho, camera_dir=c_dir, camera_pos = c_pos, fov = 50)

    vp.zoom_all(size = fig_size)


    vp.camera_dir = c_dir
    data.cell.vis.enabled = False
    renderer = TachyonRenderer(ambient_occlusion_brightness=1, depth_of_field=True)

    for frame in frames:
        outfile = f"{outfile_prefix}_{orient}_frame{frame}.png"
        if orientation == (1, 1, 0) and frame == 0:
            continue
        print(f"rendering {outfile} ..")
        vp.render_image(filename=outfile, frame=frame, size=fig_size, alpha=alpha, background=background, renderer=renderer)

    pipeline.remove_from_scene()


@dataclass
class Dumpfile:
    path: str
    frames: list



class Simulation:
    def __init__(self, dumpfiles: tuple, size: int, radius: int, sim_num: int, in_dir: str = "dumpfiles", out_dir: str = "figs"):
        self.dumpfiles = dumpfiles
        self.size = size
        self.radius = radius
        self.sim_num = sim_num
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.out_prefix  = f"{self.sim_num}_n{self.size}_r{self.radius}"



    def render(self):
        orientations = [(1, 0, 0), (1, 1, 0)]
        for dump in self.dumpfiles:
            for ori in orientations:
                render_slice(f"{self.in_dir}/{dump.path}", f"{self.out_dir}/{self.out_prefix}", dump.frames, orientation=ori)




def main():
    # fname = "/home/alexander/compsci/thesis/results/pore_faceting/dumpfiles/1_n90_r40_10ns_200ns.dump"
    # out = "/home/alexander/compsci/thesis/results/pore_faceting/dumpfiles/n90_r40"

    dump1 = Dumpfile("1_n90_r40_10ns.dump", 0)
    dump2 = Dumpfile("1_n90_r40_10ns_200ns.dump", 74)
    sim1 = Simulation((dump1, dump2), 90, 40, 1)
    # sim1.out_prefix = f"TEST_{sim1.out_prefix}"
    # sim1.render()
    # return

    dump1 = Dumpfile("2_n100_r40_10ns.dump", 0)
    dump2 = Dumpfile("2_n100_r40_25ns_100ns.dump", 96)
    sim2 = Simulation((dump1, dump2), 100, 40, 2)
    # sim2.render()

    dump1 = Dumpfile("3_t0.data", 0)
    dump2 = Dumpfile("3_n75_r25_100ns.dump", 987)
    sim3 = Simulation((dump1, dump2), 75, 25, 3)
    # sim3.render()

    dump1 = Dumpfile("4_t0.data", 0)
    dump2 = Dumpfile("4_n50_r15_200ns.dump", 1051)
    # dump2 = Dumpfile("4_n50_r15_201ns.dump", 5139)
    sim4 = Simulation((dump1, dump2), 50, 15, 4)
    sim4.render()

    dump1 = Dumpfile("5_n112_r40_50ns.dump", (100))
    sim5 = Simulation([dump1], 112, 40, 5)
    # sim5.render()








if __name__ == '__main__':
    main()
























    #

from ovito.modifiers import *
from ovito.vis import *
from ovito.io import *
from PySide2 import QtCore
import numpy as np
import math
from dataclasses import dataclass
from pathlib import Path


def render_particles(infile, outdir, frames, orientation=(1,1,0), particle_radius = 0.4, fig_size=(800,600), background=(0,0,0), alpha=1, annotation=False):
    """
    Renders particles and surface mesh

    from ovito.vis import TextLabelOverlay, Viewport
    from PySide2 import QtCore

    # Create the overlay:
    overlay = TextLabelOverlay(
        text = 'Some text',
        alignment = QtCore.Qt.AlignHCenter ^ QtCore.Qt.AlignBottom,
        offset_y = 0.1,
        font_size = 0.03,
        text_color = (0,0,0))
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
    else:
        raise NotImplementedError


    pipeline.modifiers.append(IdentifyDiamondModifier())
    #pipeline.modifiers.append(SliceModifier(normal=orientation, distance=d_min, inverse=False))
    sm = ConstructSurfaceModifier(
        radius = 2.4000000000000004,
        smoothing_level = 6,
        radius_scaling = 0.9400000000000002,
        isolevel = 0.1)
    pipeline.modifiers.append(sm)
    # pipeline.modifiers.append(SliceModifier(normal=orientation, distance=d_max, inverse=False))
    #pipeline.modifiers.append(AmbientOcclusionModifier(intensity=0.6))

    data = pipeline.compute()
    data.particles.vis.radius = particle_radius


    pipeline.add_to_scene()
    # if system_size is None:
    vp = Viewport(type = Viewport.Type.Ortho, camera_dir=orientation)
    # if annotation:
    #     overlay = TextLabelOverlay(
    #         text = "a)",
    #         alignment = QtCore.Qt.AlignHCenter ^ QtCore.Qt.AlignBottom,
    #         offset_y = 0.7,
    #         offset_x = -0.3,
    #         font_size = 0.06,
    #         text_color = (0,0,0))
    #
    #     vp.overlays.append(overlay)


    # breakpoint()
    # else:
    c_pos = tuple([size/2]*3)
    if orientation == (1, 1, 0):
        c_dir = (0.7, 0.7, 0)
    elif orientation == (1, 0, 0):
        c_dir = (-1, 0, 0)
    # vp = Viewport(type = Viewport.Type.Ortho, camera_dir=c_dir, camera_pos = c_pos, fov = 50)

    vp.zoom_all(size = fig_size)

    vp.camera_dir = c_dir
    data.cell.vis.enabled = False
    # breakpoint()
    pipeline.vis_elements[2].reverse_orientation=True
    pipeline.vis_elements[2].surface_color=(85/255, 255/255, 127/255)
    # pipeline.vis_elements[2].surface_color=(0.5, 0.5, 0.5)
    renderer = OpenGLRenderer()

    if particle_radius != 0.0:
        pp = 1
    else:
        pp = 0

    annot = ["a", "b", "c", "d"]

    i = 0
    for frame in frames:
        if particle_radius == 0.0 and annotation:
            overlay = TextLabelOverlay(
                text = annot[i],
                alignment = QtCore.Qt.AlignHCenter ^ QtCore.Qt.AlignBottom,
                offset_y = 0.65,
                offset_x = 0.15,
                font_size = 0.1,
                text_color = (0,0,0))
            i += 1
            vp.overlays.append(overlay)



        outfile = Path(outdir) / f"p{pp}_frame{frame}.png"
        print(f"rendering {outfile} ..")
        vp.render_image(filename=str(outfile), frame=frame, size=fig_size, alpha=alpha, background=background, renderer=renderer)
        if particle_radius == 0.0 and annotation:
            vp.overlays.remove(overlay)

    pipeline.remove_from_scene()

def main():
    fname = "/home/alexander/compsci/thesis/dev/SiC_inverted_crystals/deform/proper_init_temp/completed/T2200_100ns_scale0.85/trajectories/traj_100ns.dump"
    outpath = "figs/"
    frames = [0, 906, 907, 2000]
    # frames = [0, 137, 138, 290]
    # frames = [0, 150, 152, 330]
    for particle_radius in (0.0, 0.4):
        if particle_radius == 0.0:
            annotation = True
        render_particles(fname, outpath, frames, particle_radius=particle_radius, annotation=annotation)

if __name__ == '__main__':
    main()

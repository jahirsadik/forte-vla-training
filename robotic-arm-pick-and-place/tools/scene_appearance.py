"""Original procedural finishes inspired by the supplied lab and shelf photos.

Decorations have no collision masks or mass. Textures are authored here rather
than embedding photographs, so rebuilding requires no external image assets.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_textures(destination):
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(27)
    y, x = np.mgrid[0:1024, 0:1024] / 1024
    warp = x + 0.018 * np.sin(y * 8) + 0.008 * np.sin(y * 23 + x * 5)
    grain = (
        7 * np.sin(2 * np.pi * (warp * 32 + 0.5 * np.sin(warp * 9)))
        + 3 * np.sin(2 * np.pi * warp * 103)
        + 2 * np.sin(2 * np.pi * warp * 251)
        + 10 * np.sin(warp * 13 + y * 2)
        + rng.normal(0, 1.5, x.shape)
    )
    wood = np.clip([184, 128, 72] + grain[..., None] * [1, 0.82, 0.57], 0, 255)
    Image.fromarray(wood.astype("uint8")).save(destination / "wood.png")

    noise = rng.normal(0, 1.4, x.shape) + 1.3 * np.sin(x * 20 + y * 9)
    top = Image.fromarray(
        np.clip([43, 44, 43] + noise[..., None], 0, 255).astype("uint8")
    )
    draw = ImageDraw.Draw(top)
    draw.rectangle((18, 18, 1005, 1005), outline=(219, 218, 205), width=9)
    # Restrained layout markings and wear on the black laboratory work surface.
    for p in (256, 512, 768):
        for q in range(70, 960, 58):
            draw.line((p, q, p, q + 14), fill=(83, 84, 79), width=1)
            draw.line((q, p, q + 14, p), fill=(83, 84, 79), width=1)
    for _ in range(80):
        u, v = rng.integers(45, 975, 2)
        du, dv = rng.integers(-15, 16, 2)
        draw.line((int(u), int(v), int(u + du), int(v + dv)), fill=(56, 57, 55))
    top.save(destination / "worktop.png")

    paper_noise = rng.normal(0, 1, x.shape) + 3 * np.sin(y * 2 * np.pi * 260)
    paper = np.clip([229, 223, 203] + paper_noise[..., None], 0, 255).astype("uint8")
    Image.fromarray(paper).save(destination / "paper.png")
    stone_noise = rng.normal(0, 1.6, x.shape) + 2 * np.sin(x * 18) * np.sin(y * 21)
    floor = Image.fromarray(
        np.clip([184, 178, 163] + stone_noise[..., None], 0, 255).astype("uint8")
    )
    ImageDraw.Draw(floor).rectangle(
        (1, 1, 1022, 1022), outline=(142, 138, 128), width=4
    )
    floor.save(destination / "floor.png")


def spine_texture(destination, book, index):
    """Printed cloth-bound spine, with original titles and retained task colors."""
    rng = np.random.default_rng(100 + index)
    color = np.array(book["color"]) * 210 + 22
    grain = rng.normal(0, 2, (1024, 192, 1))
    spine = Image.fromarray(np.clip(color + grain, 0, 255).astype("uint8"))
    draw = ImageDraw.Draw(spine)
    ink = (233, 223, 192)
    draw.rectangle((5, 5, 186, 1018), outline=tuple((color * 0.7).astype(int)), width=4)
    for z in (75, 86, 842, 853):
        draw.line((15, z, 176, z), fill=(194, 166, 100), width=3)
    title = book.get("title", book["name"].upper() + " FIELD NOTES")
    text = Image.new("RGBA", (740, 80), (0, 0, 0, 0))
    ImageDraw.Draw(text).text(
        (370, 40), title, font=ImageFont.load_default(size=56), fill=ink, anchor="mm"
    )
    text = text.rotate(-90, expand=True)
    spine.paste(text, ((192 - text.width) // 2, 98), text)
    draw.text(
        (96, 38), "FORTE", font=ImageFont.load_default(size=24), fill=ink, anchor="mm"
    )
    draw.rounded_rectangle((35, 891, 157, 982), radius=4, fill=(224, 215, 187))
    draw.text(
        (96, 915),
        "LAB",
        font=ImageFont.load_default(size=22),
        fill=(43, 44, 42),
        anchor="mm",
    )
    draw.text(
        (96, 953),
        f"{index + 1:02d}",
        font=ImageFont.load_default(size=30),
        fill=(43, 44, 42),
        anchor="mm",
    )
    filename = f"spine_{book['name']}.png"
    spine.save(destination / filename)
    return filename


def apply_lab_style(root, table, cfg, element):
    """Finish existing surfaces and add cosmetic frame/whiteboard details."""
    asset, world = root.find("asset"), root.find("worldbody")
    asset.find("material[@name='worktop']").attrib.update(
        texuniform="false", texrepeat="1 1", specular="0.18", shininess="0.18"
    )
    asset.find("material[@name='metal']").attrib.update(
        rgba="0.79 0.8 0.79 1", specular="0.45", shininess="0.35"
    )
    for name in ("wall", "wall_mat"):
        material = asset.find(f"material[@name='{name}']")
        if material is not None:
            material.set("rgba", "0.86 0.85 0.81 1")
    element(asset, "texture", name="lab_floor_tex", type="2d", file="floor.png")
    for name in ("floor", "floor_mat"):
        material = asset.find(f"material[@name='{name}']")
        if material is not None:
            material.attrib.update(
                rgba="1 1 1 1",
                texture="lab_floor_tex",
                texuniform="true",
                texrepeat="2 2",
                specular="0.08",
            )
    element(
        asset,
        "material",
        name="lab_whiteboard",
        rgba=[0.91, 0.92, 0.89, 1],
        specular=0.3,
        shininess=0.25,
    )
    element(
        asset,
        "material",
        name="lab_dark_trim",
        rgba=[0.14, 0.15, 0.15, 1],
        specular=0.25,
    )
    root.find("visual/headlight").attrib.update(
        ambient="0.32 0.32 0.32", diffuse="0.35 0.35 0.35", specular="0.4 0.4 0.4"
    )
    root.find("visual/quality").set("offsamples", "4")
    for light in world.findall("light"):
        light.set("specular", "0.5 0.5 0.5")
    visual = dict(contype=0, conaffinity=0, density=0, group=1)
    length, width = cfg["table"]["size_xy_m"]
    z = cfg["table"]["surface_z_m"]
    # Brushed perimeter fascia beneath the existing laminate top.
    for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
        pos = [0, 0, z - 0.023]
        pos[axis] = sign * ((length if axis == 0 else width) / 2 - 0.002)
        size = [length / 2, width / 2, 0.014]
        size[axis] = 0.0025
        element(
            table,
            "geom",
            name=f"lab_edge_{axis}_{sign}",
            type="box",
            pos=pos,
            size=size,
            material="metal",
            **visual,
        )
    for sign in (-1, 1):
        element(
            table,
            "geom",
            name=f"lab_frame_rail_{sign}",
            type="box",
            pos=[0, sign * (width / 2 - 0.06), z - 0.07],
            size=[length / 2 - 0.06, 0.018, 0.025],
            material="metal",
            **visual,
        )
        for x in (-length / 2 + 0.06, length / 2 - 0.06):
            element(
                table,
                "geom",
                type="cylinder",
                pos=[x, sign * (width / 2 - 0.0785), z - 0.07],
                size=[0.004, 0.0005],
                quat=[0.70710678, 0.70710678, 0, 0],
                material="lab_dark_trim",
                **visual,
            )
    # A quiet lab backdrop, mounted on the existing rear wall outside the task.
    element(
        world,
        "geom",
        name="lab_whiteboard_face",
        type="box",
        pos=[1.653, 0, 1.55],
        size=[0.004, 0.82, 0.40],
        material="lab_whiteboard",
        **visual,
    )
    for sign in (-1, 1):
        element(
            world,
            "geom",
            type="box",
            pos=[1.647, 0, 1.55 + sign * 0.407],
            size=[0.008, 0.835, 0.007],
            material="metal",
            **visual,
        )
        element(
            world,
            "geom",
            type="box",
            pos=[1.647, sign * 0.828, 1.55],
            size=[0.008, 0.007, 0.40],
            material="metal",
            **visual,
        )
    element(
        world,
        "geom",
        name="lab_marker_tray",
        type="box",
        pos=[1.625, 0, 1.135],
        size=[0.025, 0.75, 0.006],
        material="metal",
        **visual,
    )

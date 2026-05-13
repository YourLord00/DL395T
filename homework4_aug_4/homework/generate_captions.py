from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info
import json
import tqdm


def generate_all(split: str = "train"):
    data_dir = Path(__file__).parent.parent / "data" / split
    info_files = sorted(data_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} scenes in {split}")

    for info_file in tqdm.tqdm(info_files):
        scene_id = info_file.stem.replace("_info", "")
        scene_captions = []

        with open(info_file) as f:
            num_views = len(json.load(f)["detections"])

        for view_index in range(num_views):
            captions = generate_caption(str(info_file), view_index)
            image_file = f"{split}/{scene_id}_{view_index:02d}_im.jpg"
            for caption in captions:
                scene_captions.append({
                    "caption": caption,
                    "image_file": image_file,
                })

        out_file = data_dir / f"{scene_id}_captions.json"
        with open(out_file, "w") as f:
            json.dump(scene_captions, f)

    print(f"Done. Wrote {len(info_files)} captions files")


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    captions = []
    # reuse form generate_qa.py to extract necessary information for caption generation
    track_name = extract_track_info(info_path)
    karts_in_view = extract_kart_objects(info_path, view_index, img_width, img_height)

    if not karts_in_view:
        return []
    
    ego = next(k for k in karts_in_view if k["is_center_kart"])
    ego_cx, ego_cy = ego["center"]
    
    # 1. Ego car
    # {kart_name} is the ego car.
    captions.append(f"{ego['kart_name']} is the ego car.")

    # 2. Counting
    # There are {num_karts} karts in the scenario.
    captions.append(f"There are {len(karts_in_view)} karts in the scene.")

    # 3. Track name
    # The track is {track_name}.
    captions.append(f"The track is {track_name}.")

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    for k in karts_in_view:
        if k is ego:
            continue
        name = k["kart_name"]
        cx, cy = k["center"]

        if cy < ego_cy:
            captions.append(f"{name} is in front of the ego car.")
        else:
            captions.append(f"{name} is behind the ego car.")

        if cx < ego_cx:
            captions.append(f"{name} is left of the ego car.")
        else:
            captions.append(f"{name} is right of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_all})


if __name__ == "__main__":
    main()

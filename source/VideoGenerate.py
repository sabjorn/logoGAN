import os
import glob
import argparse
import subprocess as sp
from datetime import date

from utilities.utilities import interpolate_points, load_model
from CVAE import CVAE, Sampling
from DataGenerator import DataGenerator

from PIL import Image
import numpy as np
from tensorflow import keras

def createVideo(network, coordinates, total_time_s, keyframe_hold_time_s, framerate, output_file):
    # calculate keyframe info
    num_images = len(coordinates)

    seconds_per_section = (total_time_s - (num_images * keyframe_hold_time_s)) / num_images
    n_steps = int(framerate * seconds_per_section)
    if(n_steps < 1):
        raise Exception('n_steps is less than 1')

    keyframe_hold_time_f = int(keyframe_hold_time_s * framerate)

    input_shape = network.input_shape
    latent_dim = network.latent_dim

    command = [ "ffmpeg",
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-framerate', f"{framerate}", # frames per second
            '-s', f"{input_shape[0]}x{input_shape[1]}", # size of one frame
            '-i', '-', # The imput comes from a pipe
            '-framerate', f"{framerate}", # frames per second
            '-an', # Tells FFMPEG not to expect any audio
            f"{output_file}" ]
    pipe = sp.Popen(command, stdin=sp.PIPE)

    first_coords = coordinates[0]
    for coord in coordinates[1:]:
        keyframe = network.generate_image(np.expand_dims(first_coords, axis=0))
        for hold_frame in range(keyframe_hold_time_f):
            pipe.stdin.write(keyframe.tostring())
        
        interpolated = interpolate_points(first_coords, coord, n_steps=n_steps)
        for i, frame in enumerate(interpolated):
            image = network.generate_image(np.expand_dims(frame, axis=0))
            if(image.ndim > 3):
                image = image[0]
            pipe.stdin.write(image.tostring())
        first_coords = coord

    # help make video loop smooth
    keyframe = network.generate_image(np.expand_dims(first_coords, axis=0))
    pipe.stdin.write(keyframe.tostring())
    pipe.stdin.write(keyframe.tostring())
    pipe.stdin.write(keyframe.tostring())
    
    pipe.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates a video from a Keras model.")
    parser.add_argument(
        "model_dir",
        help="directory containing keras models",
        type=str)
    parser.add_argument(
        "model_id",
        help="hash of the model",
        type=str)
    parser.add_argument(
        "epoch",
        help="epoch (i.e. 'epoch80')",
        type=str)
    parser.add_argument(
        "img_dir",
        help="location of img directory",
        type=str)
    parser.add_argument(
        "out_dir",
        help="output dir for video",
        type=str)
    parser.add_argument(
        "-f",
        "--framerate",
        help="framerate of output video",
        type=float,
        default=60.0)
    parser.add_argument(
        "-t",
        "--holdtime",
        help="time (in seconds) that a keyframe is held for",
        type=float,
        default=1.0)
    parser.add_argument(
        "-l",
        "--length",
        help="target length of the output video (in seconds)",
        type=int,
        default=5)
    parser.add_argument(
        "-b",
        "--background",
        help="set backround colour for image scaling, must take the form '255 255 255'",
        nargs="+",
        type=int)
    args = parser.parse_args()

    CONFIG = {
        "model_id": args.model_id, 
        "model_epoch": args.epoch, 
        "img_directory": args.img_dir, 
        "output_dir": args.out_dir,
        "model_dir": args.model_dir,
        "background_color": tuple(args.background or [255, 255, 255]),
        "framerate": args.framerate,
        "keyframe_hold_time_s": args.holdtime, # seconds to hold on frame
        "total_time_s": args.length # total video time
    }

    encoder_file = os.path.join(CONFIG['model_dir'], f"encoder_at_{CONFIG['model_epoch']}.h5")
    decoder_file = os.path.join(CONFIG['model_dir'], f"decoder_at_{CONFIG['model_epoch']}.h5")

    # check each image exists and can be opened
    # convert to points and print to file
    list_of_images = glob.glob(os.path.join(CONFIG['img_directory'], "*.*"))
    num_images = len(list_of_images)

    encoder = keras.models.load_model(encoder_file, {"Sampling":Sampling})

    sequence = []
    for img in list_of_images:
        try:
            p_img = Image.open(img)
        except Exception as e:
            print("error opening image")
            raise(e)

        input_dims = encoder.input_shape[1:]
        input_img = DataGenerator.prepare_image(p_img, input_dims, convert=None, background_color=CONFIG['background_color'])
        # get latentspace position
        _, _, z = encoder(np.expand_dims(input_img, 0))
        sequence.append({"img_name": img, "coordinates": z})
    print("images loaded")

    ## Generate video
    coordinates = [img["coordinates"].numpy()[0] for img in sequence]
    coordinates.append(coordinates[0]) # circular list

    video_name = f"HEARTA_{CONFIG['model_id']}_{CONFIG['model_epoch']}_f{CONFIG['framerate']}_khold{CONFIG['keyframe_hold_time_s']}_ttime{CONFIG['total_time_s']}.mp4"
    output_file = os.path.join(CONFIG['output_dir'], video_name)
    model = load_model(decoder_file)
    createVideo(model, coordinates, CONFIG['total_time_s'], CONFIG['keyframe_hold_time_s'], CONFIG['framerate'], output_file)

    ## Generate "record" of sequence
    filename = f"{CONFIG['model_id']}_{CONFIG['model_epoch']}_generated_video_record"
    filename_path = os.path.join(CONFIG['output_dir'], filename + ".txt")
    with open(filename_path, "w") as f:
        f.write(f"date: {date.today()}\n")
        f.write(f"{CONFIG}\n")

        f.write("model sequence = [")
        for seq in sequence:
            f.write(f"{seq}, \n")
        f.write("]")

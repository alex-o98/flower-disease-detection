from argparse import ArgumentParser,ArgumentTypeError



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c","--camera",
                        help = "Insert if you want to use the camera or not. [0 / 1 , default=0]",
                        type=int,default=0)

    parser.add_argument("-cn","--camera_number",
                        help = "Insert the camera number. If you have a camera inserted and you can't \
                                see any output or you want to use another camera, try changing this number \
                                to another value [0 - 10, default=0]",
                        type=int,default=0)
                        
    parser.add_argument("-v","--video_path",
                        help = "Insert the path to a video", 
                        type=str)

    parser.add_argument("-i","--image_path",
                        help = "Insert the path to an image", 
                        type=str)
                        
    parser.add_argument("-vc","--version_cascade",
                        help = "Which version of cascade you want to use [1 - 7 , default=5]",
                        default = 5, type=int)

    parser.add_argument("-dt","--disease_threshold",
                        help = "The maxmum number of diseases found on a leaf until it is \
                        considered healthy [1-20 , default=2]",                    
                        default = 2, type=float)

    arguments = parser.parse_args()

    parsed_arguments = {}

    # Parsing the arguments    
    for argc in vars(arguments):        
        parsed_arguments[argc] = getattr(arguments,argc)

    cam = cn = img = vid = vc = dt = None

    # Check whether to use the camera or not
    if parsed_arguments['camera'] < 0 or parsed_arguments['camera'] > 1:
        raise ArgumentTypeError("Select either 0 (No camera) or 1 (Using a camera)")
    else:
        cam = parsed_arguments['camera']
        if parsed_arguments['camera_number'] < 0 or parsed_arguments['camera_number'] > 10:
            raise ArgumentTypeError("Select either 0 (No camera) or 1 (Using a camera)")
        else:
            cn = parsed_arguments['camera_number']

    # Cascade version    
    if parsed_arguments['version_cascade'] < 1 or parsed_arguments['version_cascade'] > 7:
        raise ArgumentTypeError("Select a version between 1 and 7")
    else:
        vc = parsed_arguments['version_cascade']

    # Box difference 
    if parsed_arguments['disease_threshold'] < 1 or parsed_arguments['disease_threshold'] > 20:
        raise ArgumentTypeError("Select a disease threshold between 1 and 20")
    else:
        dt = parsed_arguments['disease_threshold']
    
    vid = "testing/1.mp4"    
    cam=1
    cn=0

    if cam:
        print("Using cam")
        from leaf_segmentation import detect_diseases_camera
        detect_diseases_camera(cn,vc,dt)
    elif img is not None:    
        from leaf_segmentation import detect_diseases_image
        detect_diseases_image(img,vc,dt)

    elif vid is not None:       
        from leaf_segmentation import detect_diseases_video        
        detect_diseases_video(vid,vc,dt)

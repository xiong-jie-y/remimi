from remimi.generators.parallax_video_generator import ParallaxVideoGeneratorOption


looking_glass_8_9_inpaint_option = ParallaxVideoGeneratorOption(
    video_input_width = 820, 
    video_input_height = 460,
    stereo_baseline = 0.00000140,
    focal_length=2000.0,
    inpaint=True,
    foreground_forward_z=0.7
)

DEFAULT_OPTIONS = {
    "looking_glass_8_9_inpaint_option": looking_glass_8_9_inpaint_option,
}
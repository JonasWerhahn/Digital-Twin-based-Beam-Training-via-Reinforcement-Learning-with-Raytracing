import imageio.v2 as imageio
import bpy
import torch
import os
from advanced_env import BlenderEnv
import math
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils import stdout_redirected

# reads log of rl agent trained on blender environment
def read_log(name):
    log_actions = []
    log_observations = []
    log_pixels = []
    log_brightnesses = []
    log_distances = []
    log_rewards = []

    log = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Results/{name}_log.txt"), "r")
    lines = log.readlines()
    log.close()

    lines.pop(0)
    lines.pop(0)

    current_episode = 0

    # reads actions and observation of every step of certain episode
    for line in lines:
        if line.rstrip():
            if current_episode==episode:
                print(line)
                values = line.split()
                log_actions.append([float(values[0]), float(values[1])])
                log_observations.append([float(values[2]), float(values[3])])
                log_pixels.append([int(values[4]), int(values[5])])
                log_brightnesses.append(int(float(values[6])))
                log_distances.append(float(values[7]))
                log_rewards.append(int(float(values[8])))
        else:
            if current_episode==episode: break
            else: current_episode += 1
            
    return log_actions, log_observations, log_pixels, log_brightnesses, log_distances, log_rewards

# draws vision area of agent onto animation frames
def draw_circle_mask(img):
    yy, xx = torch.meshgrid(
        torch.arange(new_res), torch.arange(new_res), indexing='ij'
    )

    center = new_res / 2
    radius = 0.5/angle_factor*new_res

    distances = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    mask = torch.abs(distances - radius)<1.0

    img[:, mask] = torch.tensor([0.99, 0.5, 0.15]).view(3, 1)  # orange pixel on border of agent vision field

# keyframe camera movement by the agent
def make_keyframes(log_actions):
    for i, action in enumerate(log_actions):
        bpy.context.scene.frame_set(first_frame+i)
        current_rotation = env.camera.rotation_euler
        env.camera.rotation_euler[0] = math.radians((math.degrees(current_rotation[0]) + action[0]))
        env.camera.rotation_euler[2] = math.radians((math.degrees(current_rotation[2]) + action[1])%360)
            
        # enforce camera bounds
        env.camera.rotation_euler[0] = max(env.camera.rotation_euler[0], 0.0)
        env.camera.rotation_euler[0] = min(env.camera.rotation_euler[0], math.radians(80))
        
        # keyframe rotation for animation
        env.camera.keyframe_insert(data_path="rotation_euler", index=-1)

# renders image of environment of certain frame
def get_render_image(frame):
    print(f"rendering frame {frame}")
    bpy.context.scene.frame_set(frame)
    with stdout_redirected():
        bpy.ops.render.render(write_still=True) # render image
        
    # read rendered image
    image = Image.open(render_path).convert("RGB")
    transform = transforms.ToTensor()
    return transform(image)

# draws position of observed brightest pixel onto animation frames
def draw_brightest_pixel(x, y):
    start = new_res*(1-1/angle_factor)/2
    res_factor = new_res/200/2
    target_x = int(start + x*res_factor)
    target_y = int(start + y*res_factor)
    thickness = 5 # thickness of drawed position
    
    for a in range(thickness*thickness):
        if color_env: tensor[:, target_x+a//thickness-thickness//2, target_y+a%thickness-thickness//2] = torch.tensor([0.0, 0.99, 0.0])
        else: tensor[:, target_x+a//thickness-thickness//2, target_y+a%thickness-thickness//2] = torch.tensor([0.99, 0.0, 0.0])

# draws debug information on each animated frame
def draw_information(arr, ep, frame, max_frame, next_action, observation, pixel, brightness, dist_to_target, reward, target):
    font_size = int(new_res/40)
    gap = int(font_size*1.2)
    
    pil_img = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    
    draw.text((5, 2), f"Episode {ep}, Frame {frame}/{max_frame}", font=font, fill=(255,255,255)) # number of episode, number of frame
    draw.text((5, gap+2), f"Target: {target}", font=font, fill=(255,255,255)) # target type
    draw.text((5, gap*2+2), f"Observation {round(observation[0]*80, 2)}°, {round(observation[1]*360, 2)}°", font=font, fill=(255,255,255)) # current orientation
    draw.text((5, gap*3+2), f"Next Action {next_action}", font=font, fill=(255,255,255)) # performed action for next frame
    draw.text((5, gap*4+2), f"Brightest Pixel {pixel[0]}, {pixel[1]}, Brightness {brightness}", font=font, fill=(255,255,255)) # position of brightest pixel
    draw.text((5, gap*5+2), f"Distance to Target {round(dist_to_target, 2)}", font=font, fill=(255,255,255)) # distance camera-to-target
    draw.text((5, gap*6+2), f"Reward {reward}", font=font, fill=(255,255,255)) # gained reward

    return np.array(pil_img)


episode = 23949
agent = "adv_ddpg_hl4nf05"
color_env = True # show environment with original colors
bright_background = True # brigthen the background for better visibility
save_blend_file = False # save blend file with animated camera movement
draw_circle = True # draws area of seen pixels by the agent
draw_pixel = True # emphasizes the brightest observed pixel
show_information = True # shows debug information on each frame
transparent_target = True # shows target if it would be obstructed by a building
new_res = 800 # reselution of animation
angle_factor = 2.0 # bigger fov of animation

log_actions, log_observations, log_pixels, log_brightnesses, log_distances, log_rewards = read_log(agent)
if color_env: env = BlenderEnv(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), "ignore.txt"), agent, "advanced_color.blend")
else: env = BlenderEnv(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), "ignore.txt"), agent, "advanced_env.blend")
env.read_env_variations()
env.number_resets = episode
env.reset()
first_frame = env.current_frame
mesh = bpy.data.objects.get("Google 3D Tiles")

make_keyframes(log_actions)

env.camera.data.angle *= angle_factor

bpy.context.scene.render.resolution_x = new_res
bpy.context.scene.render.resolution_y = new_res

if bright_background:
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 0.3

render_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Renders/animation_render.png")
bpy.context.scene.render.filepath = render_path
    
if save_blend_file:
    bpy.ops.wm.save_as_mainfile(filepath=f"{env.env_path}/{agent}replay_animation{episode}.blend")

frames = []
for i in range(len(log_actions)):
    mesh.hide_render = False
    tensor = get_render_image(first_frame+i)
    if transparent_target:
        transparency = 0.4
        mesh.hide_render = True
        tensor2 = get_render_image(first_frame+i)
        tensor = tensor*(1-transparency)+tensor2*transparency

    if draw_circle: draw_circle_mask(tensor)
    if draw_pixel and log_brightnesses[i]>=10: draw_brightest_pixel(log_pixels[i][0], log_pixels[i][1])

    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    
    if show_information: 
        next_action = ""
        if i+1 < len(log_actions):
            arrows = "↙↓↘←.→↖↑↗"
            index = 0

            if abs(log_actions[i+1][0]) < 1.0: index += 3
            elif log_actions[i+1][0] > 0.0: index += 6
            if abs(log_actions[i+1][1]) < 1.0: index += 1
            elif log_actions[i+1][1] < 0.0: index += 2

            next_action += arrows[index]
            next_action += f" {round(log_actions[i+1][0], 2)}, {round(log_actions[i+1][1], 2)}"

        target = env.target_variations[episode][0]
        max_frame = first_frame+len(log_observations)-1
        arr = draw_information(arr, episode, first_frame+i, max_frame, next_action, log_observations[i], log_pixels[i], log_brightnesses[i], log_distances[i], log_rewards[i], target)

    frames.append(arr)
    
color = "_color" if color_env else ""

with imageio.get_writer(f"../{agent}_ep{episode}{color}.mp4", fps=0.5) as writer:
    for f in frames:
        writer.append_data(f)
import numpy as np
import bpy
import math
import os
import torch
from torchvision import transforms
from PIL import Image
from utils import stdout_redirected


class BlenderEnv():
    def __init__(self, training_episodes, log_path, agent_type, env_name = "advanced_env.blend"):
        # render parameters
        self.env_render_engine = 'CYCLES'
        self.render_resolution_x = 200
        self.render_resolution_y = 200
        
        self.env_name = env_name
        self.env_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../Blender")
        self.current_frame = 0
        self.frame_end = 40 # maximum number of frames in traffic simulation
        self.training_episodes = training_episodes
        self.log_file = open(log_path, "w")
        self.agent_type = agent_type
        
        self.first_reset = True
        
        file_path = os.path.join(self.env_path, self.env_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Blend-Datei nicht gefunden: {file_path}")
        
        bpy.ops.wm.open_mainfile(filepath=file_path)
        
        if not bpy.data.scenes:
            raise RuntimeError("Keine Szenen in der geladenen Blend-Datei gefunden.")

        if bpy.context.scene is None:
            bpy.context.window.scene = bpy.data.scenes[0]
        
        bpy.context.scene.render.engine = self.env_render_engine
        bpy.context.scene.render.resolution_x = self.render_resolution_x
        bpy.context.scene.render.resolution_y = self.render_resolution_y
        bpy.data.scenes[0].render.engine = "CYCLES"
        
        # lst possible frame
        bpy.context.scene.frame_end = self.frame_end

        self.camera = bpy.data.objects["Cam"]
        
        self.render_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Renders/{self.agent_type}_agent.png")
        bpy.context.scene.render.filepath = self.render_path
        bpy.context.scene.camera = self.camera
        self.camera_pos = self.camera.matrix_world.translation
        
        self.pedestrian = bpy.data.objects["Pedestrian"]
        self.cyclist = bpy.data.objects["Cyclist"]
        self.car = bpy.data.objects["Car"]
        self.bus = bpy.data.objects["Bus"]
        self.train = bpy.data.objects["Train"]

        # defines how common certain target types are
        self.target_type_importances = [1, 3, 23, 25, 29]
        
        self.velocity_multiplier = 4000
        
        self.pedestrian_points, self.cyclist_paths, self.car_paths, self.bus_paths, self.train_paths = [], [], [], [], []
        
        # receive possible pedestrian positions
        for i in range(79):
            self.pedestrian_points.append(bpy.data.objects[f"PedestrianPoint{i+1}"])
            
        # receive possible cyclist paths
        for i in range(3):
            path = bpy.data.objects[f"CyclePath{i+1}"]
            velocity = 4.17 #15 km/h
            path_length, importance, velocity = self.init_path(path, velocity)
            self.cyclist_paths.append([path, path_length, velocity, importance])
            
        # receive possible car paths
        for i in range(14):
            path = bpy.data.objects[f"CarPath{i+1}"]
            velocity = 8.3 #30 km/h
            path_length, importance, velocity = self.init_path(path, velocity)
            self.car_paths.append([path, path_length, velocity, importance])
            
        # receive possible bus paths
        for i in range(4):
            path = bpy.data.objects[f"BusPath{i+1}"]
            velocity = 8.3 #30 km/h
            path_length, importance, velocity = self.init_path(path, velocity)
            self.bus_paths.append([path, path_length, velocity, importance])
            
        # receive possible train paths
        for i in range(5):
            path = bpy.data.objects[f"TrainPath{i+1}"]
            velocity = 16.7 #60 km/h
            path_length, importance, velocity = self.init_path(path, velocity)
            self.train_paths.append([path, path_length, velocity, importance])
        
        
        self.antenna_angle_mask = self.create_incircle_mask() # generate relevance mask for incomming light
        
        self.number_resets = 0
        self.target_variations = []
        self.mat_variations = []
        
        self.log_file.write(f"{self.training_episodes}\n")
        
        self.last_target_pixel = [0.0, 0.0]
        self.last_max_brigthness = 0.0
    
    # resets state of environment based on random variations
    def reset(self):
        self.current_frame = 0
        # set state of traffic animation
        bpy.context.scene.frame_set(self.current_frame)

        # delete all targets
        bpy.ops.object.select_all(action='DESELECT')
        if not self.first_reset: self.target.select_set(True)
        self.first_reset = False
        bpy.ops.object.delete()
        
        # generate target based on variations
        variation = self.target_variations[self.number_resets]
        match variation[0]:
            case "Pedestrian":
                pedestrian_point = self.pedestrian_points[variation[1]]
                pedestrian_clone = self.clone_target(variation[0])
                pedestrian_clone.name = f"{variation[0]}Target"
                pedestrian_clone.location = pedestrian_point.location
                        
                pedestrian_point.location[0] += variation[2]
                pedestrian_point.location[1] += variation[3]
            case "Cyclist":
                clone = self.clone_target(variation[0])
                clone.name = f"{variation[0]}Target"
                self.let_object_follow_path(clone, self.cyclist_paths[variation[1]][0], variation)
            case "Car":
                clone = self.clone_target(variation[0])
                clone.name = f"{variation[0]}Target"
                self.let_object_follow_path(clone, self.car_paths[variation[1]][0], variation)
            case "Bus":
                clone = self.clone_target(variation[0])
                clone.name = f"{variation[0]}Target"
                self.let_object_follow_path(clone, self.bus_paths[variation[1]][0], variation)
            case "Train":
                clone = self.clone_target(variation[0])
                clone.name = f"{variation[0]}Target"
                self.let_object_follow_path(clone, self.train_paths[variation[1]][0], variation)
                
             
        # initialize materials of environment based on variations 
        obj = bpy.data.objects.get("Google 3D Tiles")
        mat_count = 0
        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            variation = self.mat_variations[self.number_resets][mat_count]
            nodes = mat.node_tree.nodes
            for node in nodes:
                if node.type == 'BSDF_DIFFUSE':
                    node.inputs["Color"].default_value = (variation[0], variation[0], variation[0], 1.0) # random diffuse brightness
                
                elif node.type == 'BSDF_GLOSSY':
                    node.inputs["Color"].default_value = (variation[1], variation[1], variation[1], 1.0) # random direct reflection brightness
                    node.inputs["Roughness"].default_value = variation[2] # random reflection roughness
            mat_count += 1
                    
        bpy.context.view_layer.update()
        self.number_resets += 1
        
        self.log_file.write("\n")
        
        while((not self.is_target_in_bounds()) and self.current_frame<self.frame_end):
            self.current_frame += 1
            # set state of traffic animation
            bpy.context.scene.frame_set(self.current_frame)
        
        self.look_at_target() # beginn episode by rotating camera to target direction
        
        return self.get_state()
    
    # advance state of environment
    def step(self, action):
        # rotate cameras
        current_rotation = self.camera.rotation_euler
        self.camera.rotation_euler[0] = math.radians((math.degrees(current_rotation[0]) + action[0]))
        self.camera.rotation_euler[2] = math.radians((math.degrees(current_rotation[2]) + action[1])%360)
        
        # enforce camera bounds
        self.camera.rotation_euler[0] = max(self.camera.rotation_euler[0], 0.0)
        self.camera.rotation_euler[0] = min(self.camera.rotation_euler[0], math.radians(80))
            
        bpy.context.view_layer.update()
            
        reward = self.get_reward()
        state = self.get_state()
        
        # print step informations
        print(f"action: {action}, rotation: {state[0]*80, state[1]*360}, angle to target: {state[2], state[3]}, last brigthness: {state[4]}, dist to target: {state[5]} reward: {reward}, target: {self.target_variations[self.number_resets-1][0]}")
        
        # log step informations
        self.log_file.write(f"{action[0]} {action[1]} {state[0]} {state[1]} {state[2]} {state[3]} {state[4]} {state[5]} {reward}\n")
        
        self.current_frame += 1
        # set state of traffic animation
        bpy.context.scene.frame_set(self.current_frame)
        
        # end episode if target leaves environment bounds or frame count exceeds max frame
        terminated = ((not self.is_target_in_bounds()) or self.current_frame>=self.frame_end)
        
        return state, reward, terminated
    
    # returns current camera rotations, position of brightest pixel of last rendered image, brightness of brightest pixel and distance camera-to-target
    def get_state(self):
        x_rot = math.degrees(self.camera.rotation_euler[0])%360
        z_rot = math.degrees(self.camera.rotation_euler[2])%360
        
        # normalize camera rotation to camera bounds
        x_rot = x_rot/80.0
        z_rot = z_rot/360.0
        
        target_direction = self.target.matrix_world.translation - self.camera_pos
        target_dist = np.sqrt(target_direction[0]**2 + target_direction[1]**2 + target_direction[2]**2)
        
        state = [x_rot, z_rot, self.last_target_pixel[0], self.last_target_pixel[1], self.last_max_brigthness, target_dist]
        return state
    
    # computes rewards based on incomming light to cameras
    def get_reward(self): 
        # render scene
        with stdout_redirected():
              bpy.ops.render.render(write_still=True)
        
        # read rendered image
        image = Image.open(self.render_path).convert("L")  # optional "L" für Graustufen
        transform = transforms.ToTensor()  # konvertiert in [0, 1] FloatTensor mit shape [C, H, W]
        tensor = transform(image)

        masked_pixels = tensor.squeeze(0)[self.antenna_angle_mask]

        # maximum brightness
        max_value = masked_pixels.max()

        masked_coords = self.antenna_angle_mask.nonzero() # every pixel inside mask
        max_coords = masked_coords[masked_pixels == max_value] # every brightest pixel inside mask

        random_index = torch.randint(len(max_coords), (1,)) # random brightest pixel inside mask
        self.last_target_pixel = max_coords[random_index].squeeze().tolist()
        self.last_max_brigthness = masked_pixels.max()*255
        
        brigthness_threshold = 10.0 # min brightness where connection between antenna and target would be possible
        
        if self.last_max_brigthness >= brigthness_threshold: return 1.0 # give reward if connection would be possible
        else: return -1.0
    
    # returns a clone of specific target type
    def clone_target(self, type):
        original = bpy.data.objects[type]
        clone = original.copy()
        clone.data = original.data.copy()
        for collection in original.users_collection:
            collection.objects.link(clone)
        self.target = clone
        return clone
    
    # animate target to move along predefined path
    def let_object_follow_path(self, object, path, variations):
        if variations[4]:
            object.rotation_euler[2] = math.pi
            
        constraint = object.constraints.new(type='FOLLOW_PATH')
        constraint.target = path
        constraint.use_curve_follow = True
        
        # controls velocity an time offset of target movement
        constraint.offset = variations[2]
        constraint.keyframe_insert(data_path="offset", frame=0)
        constraint.offset = variations[3]
        constraint.keyframe_insert(data_path="offset", frame=self.frame_end)
        
        for fc in object.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
        
        object.location[0] = variations[5]
    
    # returns length of Blenders beziercurves    
    def get_curve_length(self, curve):
        total_length = 0.0
        for spline in curve.data.splines:
            points = [point.co for point in spline.bezier_points]
            num_segments = len(points) - 1
            for i in range(num_segments):
                total_length += (points[i+1] - points[i]).length
        return total_length
    
    # defines movement on path
    def init_path(self, path, velocity):
        path_length = self.get_curve_length(path)
        importance = 1
        
        if 'velocity' in path.data:
            velocity = path.data['velocity']
        if 'importance' in path.data:
            importance *= path.data['importance']
        path.data['travel_part'] = self.velocity_multiplier*velocity/path_length
            
        path.data.eval_time = 0
        path.data.keyframe_insert(data_path="eval_time", frame=0)
        path.data.eval_time = self.velocity_multiplier*velocity/path_length
        path.data.keyframe_insert(data_path="eval_time", frame=self.frame_end)
        
        for fcurve in path.data.animation_data.action.fcurves:
            if fcurve.data_path == "eval_time":
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'LINEAR'
        
        return path_length, importance, velocity
    
    # renders environment from the top, useful to see target position
    def global_render(self, index):
        bpy.context.scene.render.resolution_x = 1700
        bpy.context.scene.render.resolution_y = 1000
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Renders/global{index}.png")
        bpy.context.scene.render.filepath = file_path
        bpy.context.scene.camera = bpy.data.objects.get("Global_Camera")
        
        bpy.ops.render.render(write_still=True)
        
        bpy.context.scene.render.resolution_x = self.render_resolution_x
        bpy.context.scene.render.resolution_y = self.render_resolution_y
        bpy.context.scene.render.engine = self.env_render_engine
        
    # create round mask based on distance to image center
    def create_incircle_mask(self):
        yy, xx = torch.meshgrid(
            torch.arange(self.render_resolution_y), torch.arange(self.render_resolution_x), indexing='ij'
        )

        # center an radius of circle
        center_x, center_y = self.render_resolution_x / 2, self.render_resolution_y / 2
        radius = min(self.render_resolution_x, self.render_resolution_y) / 2

        distances = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        # mask is true inside of circle
        mask = distances <= radius
        return mask
        
    # generates random variations to the environment for every episode
    def generate_env_variations(self):
        for ep in range(self.training_episodes):
            if ep%1000==0: print(f"Generating Episode {ep}")

            target_type = self.choose_target_type() # choose random agent targets
            match target_type:
                case 0:
                    # pedestrian
                    pedestrian_index = np.random.randint(0, len(self.pedestrian_points))
                    offset_x = np.random.uniform(high=2.0, low=-2.0)
                    offset_y = np.random.uniform(high=2.0, low=-2.0)
                    
                    self.target_variations.append(["Pedestrian", pedestrian_index, offset_x, offset_y])
                case 1:
                    # cyclist
                    path_index = self.choose_path_index(self.cyclist_paths)
                    path_variation = self.generate_path_variation(self.cyclist_paths[path_index])
                    
                    self.target_variations.append(["Cyclist", path_index, path_variation[0], path_variation[1], path_variation[2], path_variation[3]])
                case 2:
                    # car
                    path_index = self.choose_path_index(self.car_paths)
                    path_variation = self.generate_path_variation(self.car_paths[path_index])
                    
                    self.target_variations.append(["Car", path_index, path_variation[0], path_variation[1], path_variation[2], path_variation[3]])
                case 3:
                    # bus
                    path_index = self.choose_path_index(self.bus_paths)
                    path_variation = self.generate_path_variation(self.bus_paths[path_index])
                    
                    self.target_variations.append(["Bus", path_index, path_variation[0], path_variation[1], path_variation[2], path_variation[3]])
                case 4:
                    # train
                    path_index = self.choose_path_index(self.train_paths)
                    path_variation = self.generate_path_variation(self.train_paths[path_index])
                    
                    self.target_variations.append(["Train", path_index, path_variation[0], path_variation[1], path_variation[2], path_variation[3]])
            
            
            # random material variations
            obj = bpy.data.objects.get("Google 3D Tiles")
            self.mat_variations.append([])
            for _ in obj.material_slots:
                diffuse_color = np.random.uniform(0.008, 0.012) # diffusion variation
                glossy_color = np.random.uniform(0.4375, 0.5625) # variation of strength of specular highlights
                glossy_roughness = np.random.uniform(0.175, 0.225) # variation of roughness of specular highlights
                self.mat_variations[ep].append([diffuse_color, glossy_color, glossy_roughness]) # store variation
    
    # generates random variation for target movement of velocity, direction, left-to-right offset and time offset
    def generate_path_variation(self, path):
        save_margin = 6
        travel_part = self.velocity_multiplier*path[2]/path[1]
        vel_variation = np.random.uniform(0.8, 1.2)
        x_offset = np.random.uniform(high=1, low=-1)
        backwards = np.random.randint(2)==0
        
        if backwards: 
            offset = np.random.randint(((-100-travel_part)/vel_variation+save_margin), (0-save_margin))
            offset2 = offset*vel_variation+2*travel_part
        else:
            offset = np.random.randint((-100+save_margin), (travel_part/vel_variation-save_margin))
            offset2 = offset*vel_variation
                    
        return [offset, offset2, backwards, x_offset]
    
    # write variations to file
    def write_env_variations(self):
        env_variation_file = open("advanced_env_variations.txt", "w")
        env_variation_file.write(f"{self.training_episodes} episodes\n\n")
        for ep_index in range(self.training_episodes):
            env_variation_file.write(f"{self.log_list(self.target_variations[ep_index])}\n")
            env_variation_file.write(f"\n{len(self.mat_variations[ep_index])} Materials\n")
            for mat_variation in self.mat_variations[ep_index]:
                env_variation_file.write(f"{self.log_list(mat_variation)}\n")
            env_variation_file.write("\n")
        env_variation_file.close()
        
    # read variations from file    
    def read_env_variations(self):
        env_variation_file = open("advanced_env_variations.txt", "r")
        lines = env_variation_file.readlines()
        episodes = int(lines[0].split()[0])
        
        cur_line = 2
        for ep in range(episodes):
            self.target_variations.append([])
            for i, value in enumerate(lines[cur_line].strip().split()):
                if i == 1: self.target_variations[ep].append(int(value))
                elif i == 2 or i == 3 or i == 5: self.target_variations[ep].append(float(value))
                elif i == 4: self.target_variations[ep].append(bool(value))
                else: self.target_variations[ep].append(value)
                
            cur_line += 2
            
            number = int(lines[cur_line].strip().split()[0])
            cur_line += 1
            
            self.mat_variations.append([])
            for _ in range(number):
                values = lines[cur_line].strip().split()
                self.mat_variations[ep].append([float(values[0]), float(values[1]), float(values[2])])
                cur_line += 1
            cur_line += 1
        env_variation_file.close()
        
    # random target type with different chances for each target
    def choose_target_type(self):
        target_type_score = np.random.randint(0, self.target_type_importances[4])
        for i in range(len(self.target_type_importances)):
            if self.target_type_importances[i] > target_type_score:
                return i
            
    # target moves on random path
    def choose_path_index(self, paths):
        sum = 0
        path_importances = []
        for p in paths:
            sum += p[3]
            path_importances.append(sum)
            
        path_score = np.random.randint(0, sum)
        for i in range(len(path_importances)):
            if path_importances[i] > path_score:
                return i
    
    # checks if target left the simulated area
    def is_target_in_bounds(self):
        target_location = self.target.matrix_world.translation
        return ((-500 < target_location[0] < 160) and (-180 < target_location[1] < 250))
    
    # rotates camera to look at target
    def look_at_target(self):
        direction = self.target.matrix_world.translation - self.camera_pos
        rot_quat = direction.to_track_quat('-Z', 'Y')  # -Z = Blickrichtung der Kamera
        self.camera.rotation_euler = rot_quat.to_euler()
        self.camera.rotation_euler[1] = 0
            
    def log_list(self, list):
        str = ""
        for element in list:
            str += f"{element} "
        return str 
            
    def close_log_file(self):
        self.log_file.close()

# Digital-Twin-based-Beam-Training-via-Reinforcement-Learning-with-Raytracing



A method of simulating Beam Training in a realistic environment in Blender. Multiple different reinforcement agents trained in this environment. They successfully learned to keep a connection between a simulated pencil beam antenna and a moving target.

# Step-by-step procedure to create the blender environment
1. Use https://github.com/vvoovv/blosm/wiki/Import-of-Google-3D-Cities to create a 3D Mesh of a city (max 1000m x 1000m)
2. Remove Vertices to far outside of the area
3. Create BezierCurves in Blender to create paths for cars, busses, cyclists and trains
4. Create Vertices for every position a pedestrian can stand
5. Replace every material of the imported mesh by a mix-shader of Diffuse BSDF and Glossy BSDF with Fresnel blend weight
6. Create Meshes for cars, busses, cyclists, trains and pedestrians with emission shader
7. Create a camera on the postition of the antenna
8. Choose render parameters for CYCLE render-engine


Acknowledgements
Part of this work has been funded by the German Federal Ministry of Education and Research (BMBF) in the course of the 6GEM research hub under grant number 16KISK038.


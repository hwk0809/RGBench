import numpy as np
import torch

import omni.kit.commands
import omni.physxdemos as demo
import isaacsim.core.utils.prims as prims_utils
from pxr import Gf, UsdGeom,Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from isaacsim.core.api import World
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.utils.semantics import add_update_semantics, get_semantics
from isaacsim.core.utils.rotations import euler_angles_to_quat

class Deformable_Garment:
    def __init__(
        self, world:World, 
        usd_path:str="Assets/Garment/Hat/HA_Hat007/HA_Hat007_obj.usd",
        pos:np.ndarray=np.array([0.0, 0.0, 0.5]), 
        ori:np.ndarray=np.array([0.0, 0.0, 0.0]),
        scale:np.ndarray=np.array([0.0085, 0.0085, 0.0085]),
        visual_material_usd:str="assets/Assets/Material/Garment/linen_Pumpkin.usd",
        damping_scale:float=0.15, #1.0
        dynamic_friction:float=1.0,
        elasticity_damping:float=0.0,
        poissons_ratio:float=0.0,
        youngs_modulus:float=1e8,  # important parameter
        vertex_velocity_damping:float=0.0, #1.5
        sleep_damping:float=0.15,  #1.0
        sleep_threshold:float=0.15, #2.0
        settling_threshold:float=0.15, #2.0
        self_collision:bool=True,
        solver_position_iteration_count:int=16,
        simulation_hexahedral_resolution:int=24,    # important parameter
        kinematic_enabled:bool=False,
        collision_simplification:bool=True,
        collision_simplification_remeshing:bool=True,
        collision_simplification_remeshing_resolution:int=16,   # important parameter
        contact_offset:float=0.01,  # important parameter
        rest_offset:float=0.008,    # important parameter
    ):
        self.world = world
        self.usd_path=usd_path
        self.position = pos
        self.orientation = ori
        self.scale = scale
        self.visual_material_usd = visual_material_usd
        self.stage = world.stage
        self.scene = world.get_physics_context()._physics_scene
        
        self.deformable_view = UsdGeom.Xform.Define(self.stage, "/World/Deformable")
        self.deformable_name=find_unique_string_name(initial_name="deformable",is_unique_fn=lambda x: not world.scene.object_exists(x))  
        self.deformable_prim_path=find_unique_string_name("/World/Deformable/deformable",is_unique_fn=lambda x: not is_prim_path_valid(x))
        
        # define deformable object Xform
        self.deformable = SingleXFormPrim(
            prim_path=self.deformable_prim_path,
            name=self.deformable_name, 
            position=self.position,
            orientation=euler_angles_to_quat(self.orientation),
            scale=self.scale,
        )

        # add deformable object usd to stage
        add_reference_to_stage(usd_path=self.usd_path,prim_path=self.deformable_prim_path)
        
        # define deformable material for deformable object              
        self.deformable_material_path=find_unique_string_name("/World/Deformable/deformable_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.deformable_material = DeformableMaterial(
            prim_path=self.deformable_material_path,
            damping_scale=damping_scale,
            dynamic_friction=dynamic_friction,
            elasticity_damping=elasticity_damping,
            poissons_ratio=poissons_ratio,            
            youngs_modulus=youngs_modulus,
        )
        
        self.deformable_mesh_prim_path = self.deformable_prim_path+"/mesh"
        self.deformable_mesh=UsdGeom.Mesh.Get(self.stage, self.deformable_mesh_prim_path)
        self.deformable = SingleDeformablePrim(
            prim_path=self.deformable_mesh_prim_path,
            deformable_material=self.deformable_material,
            name=self.deformable_name,
            visible=True,
            vertex_velocity_damping=vertex_velocity_damping,
            sleep_damping=sleep_damping,      # glove 1.0
            sleep_threshold=sleep_threshold,    # glove 1.0
            settling_threshold=settling_threshold, # glove 1.0
            self_collision=self_collision,
            # self_collision_filter_distance=0.0,
            solver_position_iteration_count=solver_position_iteration_count,
            simulation_hexahedral_resolution=simulation_hexahedral_resolution,    # glove 72
            kinematic_enabled=kinematic_enabled,
            # -- ignore parameters below -- #
            # simulation_rest_points=None, 
            # simulation_indices=None, 
            # collision_rest_points=None,
            # collision_indices=None,
            collision_simplification=collision_simplification, 
            collision_simplification_remeshing=collision_simplification_remeshing,
            collision_simplification_remeshing_resolution=collision_simplification_remeshing_resolution,   # glove 36
            # collision_simplification_target_triangle_count=0,       
            # collision_simplification_force_conforming=False,
            # embedding=None,
            
        )
        
        # set visual material to deformable object
        if self.visual_material_usd is not None:
            self.apply_visual_material(self.visual_material_usd)
            
        self.set_contact_offset(contact_offset)  # glove 0.015
        self.set_rest_offset(rest_offset)      # glove 0.01
            
    def apply_visual_material(self,material_path:str):
        self.visual_material_path=find_unique_string_name(self.deformable_prim_path+"/visual_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(usd_path=material_path,prim_path=self.visual_material_path)
        self.visual_material_prim=prims_utils.get_prim_at_path(self.visual_material_path)
        self.material_prim=prims_utils.get_prim_children(self.visual_material_prim)[0]
        self.material_prim_path=self.material_prim.GetPath()
        self.visual_material=PreviewSurface(self.material_prim_path)
        
        self.deformable_mesh_prim=prims_utils.get_prim_at_path(self.deformable_mesh_prim_path)
        self.deformable_submesh=prims_utils.get_prim_children(self.deformable_mesh_prim)
        if len(self.deformable_submesh)==0:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.deformable_mesh_prim_path, material_path=self.material_prim_path)
        else:
            omni.kit.commands.execute('BindMaterialCommand',
            prim_path=self.deformable_mesh_prim_path, material_path=self.material_prim_path)
            for prim in self.deformable_submesh:
                omni.kit.commands.execute('BindMaterialCommand',
                prim_path=prim.GetPath(), material_path=self.material_prim_path)
                
    def set_contact_offset(self,contact_offset:float=0.01):
        self.collsionapi=PhysxSchema.PhysxCollisionAPI.Apply(self.deformable.prim)
        self.collsionapi.GetContactOffsetAttr().Set(contact_offset)
    
    def set_rest_offset(self,rest_offset:float=0.008):
        self.collsionapi=PhysxSchema.PhysxCollisionAPI.Apply(self.deformable.prim)
        self.collsionapi.GetRestOffsetAttr().Set(rest_offset)
        
    def set_garment_pose(self, pos, ori):
        if ori is not None:
            ori = euler_angles_to_quat(ori, degrees=True)
        self.deformable.set_world_pose(pos, ori)
        
    def get_garment_center_pos(self):
        return self.deformable.get_world_pose()[0]
    
    def set_mass(self,mass=0.02):
        physicsUtils.add_mass(self.world.stage, self.deformable_mesh_prim_path, mass)